"""One isolated model configuration; timed runs contain no probes or attacks."""
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
import transformers

# The language model does not use vision/audio. The NPU base environment has a
# torchvision build for a newer torch; disable those optional imports locally.
transformers.utils.import_utils._torchvision_available = False
transformers.utils.import_utils._librosa_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from linear_backends import MixedInt8Linear, NpuPool, replace_linears

ROOT = Path(__file__).resolve().parents[1]


def emit(**value):
    print(json.dumps(value), flush=True)


def memory():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value = line.partition(':')
        if key in ('VmRSS','VmHWM','VmSwap'):
            result[key+'_kib'] = int(value.split()[0])
    return result


def npu_maps():
    result, current = [], None
    for line in Path('/proc/self/smaps').read_text().splitlines():
        fields = line.split()
        if fields and '-' in fields[0] and ':' not in fields[0]:
            current = None
            if '/dev/accel/' in line:
                current = {'mapping':line}
                result.append(current)
        elif current is not None and fields and fields[0] in ('Size:','Rss:','Pss:'):
            current[fields[0][:-1]+'_kib'] = int(fields[1])
    return result


def affinity():
    allowed = os.sched_getaffinity(0)
    representatives = {}
    for cpu in sorted(allowed):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((topology/'physical_package_id').read_text().strip(),
               (topology/'core_id').read_text().strip())
        representatives.setdefault(key,cpu)
    chosen = list(representatives.values())[:4]
    assert len(chosen) == 4
    os.sched_setaffinity(0,chosen)
    return chosen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--backend',choices=['cpu','npu'],required=True)
    parser.add_argument('--dq',type=int,choices=[0,1],required=True)
    parser.add_argument('--runs',type=int,default=10)
    parser.add_argument('--warmups',type=int,default=2)
    args=parser.parse_args()
    out=ROOT/'results'/args.label
    out.mkdir(parents=True,exist_ok=True)
    cpus=affinity()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261111)
    specs=json.loads((ROOT/'prompts.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoint',local_files_only=True,trust_remote_code=False)
    for spec in specs:
        assert tokenizer.encode(spec['prompt'],add_special_tokens=False)==spec['input_ids']
        assert len(spec['input_ids'])==128
    emit(stage='loading',pid=os.getpid(),label=args.label,cpus=cpus)
    start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(ROOT/'checkpoint',torch_dtype=torch.float32,
        attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
    pool=NpuPool(128) if args.backend=='npu' else None
    replaced=replace_linears(model,bool(args.dq),pool) if args.dq or pool else []
    model.eval()
    gc.collect()
    # Release dead model-conversion/packing allocations before warmup. Apply
    # identically to all configurations; keep normal allocator caching during
    # warmup and measurement. Otherwise DQ retains freed FP32 weights in RSS.
    libc=ctypes.CDLL(None)
    libc.malloc_trim.argtypes=[ctypes.c_size_t]
    libc.malloc_trim.restype=ctypes.c_int
    libc.malloc_trim(0)
    load_s=time.perf_counter()-start
    dq_modules=[m for m in model.modules() if isinstance(m,MixedInt8Linear)]

    @torch.inference_mode()
    def request(spec,save_logits=False):
        before_calls=pool.calls if pool else 0
        before_outliers=sum(m.outlier_columns for m in dq_modules)
        begin=time.perf_counter_ns()
        inputs=tokenizer(spec['prompt'],return_tensors='pt',add_special_tokens=False)
        # New request, no shared past_key_values: no prefix/KV-cache reuse.
        output=model(**inputs,use_cache=True,logits_to_keep=1)
        logits=output.logits[0,-1]
        token=int(torch.argmax(logits).item())
        text=tokenizer.decode([token])
        end=time.perf_counter_ns()
        record={'ttft_ms':(end-begin)/1e6,'first_token':token,'first_text':text,
                'input_tokens':int(inputs['input_ids'].shape[-1]),'output_tokens':1,
                'npu_dispatches':(pool.calls-before_calls) if pool else 0,
                'outlier_columns_sum':sum(m.outlier_columns for m in dq_modules)-before_outliers}
        vector=logits.float().numpy().copy() if save_logits else None
        assert record['input_tokens']==128
        if pool:
            assert record['npu_dispatches']==168
        return record,vector

    warmups=[]
    for index in range(args.warmups):
        record,_=request(specs[0])
        warmups.append(record)
        emit(stage='warmup',index=index,**record)
    gc.collect()
    maps=npu_maps()
    mapped_kib=sum(m['Size_kib'] for m in maps)
    assert all(m['Rss_kib']==0 for m in maps), 'NPU memory accounting must avoid RSS double counting'
    if pool:
        assert maps
    (out/'smaps-ready.txt').write_text(Path('/proc/self/smaps').read_text())
    ready=memory()
    emit(stage='measured_start',monotonic_ns=time.monotonic_ns(),npu_mapped_kib=mapped_kib)
    runs=[]
    for index in range(args.runs):
        record,_=request(specs[0])
        record.update(run=index+1,memory_after=memory())
        runs.append(record)
        emit(stage='run',**record)
    emit(stage='measured_end',monotonic_ns=time.monotonic_ns())
    after=memory()
    maps_after=npu_maps()
    assert sum(m['Size_kib'] for m in maps_after)==mapped_kib
    validations=[]
    vectors={}
    for spec in specs:
        record,vector=request(spec,True)
        assert np.isfinite(vector).all()
        top=np.argsort(vector)[-5:][::-1]
        validations.append({'prompt':spec['name'],**record,
            'top_token_ids':[int(i) for i in top], 'top_logits':[float(vector[i]) for i in top]})
        vectors[spec['name']]=vector
    np.savez(out/'validation-logits.npz',**vectors)
    cpu=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
             if s.startswith('model name'))
    def version(name):
        try:return metadata.version(name)
        except metadata.PackageNotFoundError:return None
    report={'configuration':vars(args),'pid':os.getpid(),'cpu':cpu,'platform':platform.platform(),
        'cpu_affinity':cpus,'torch_threads':4,'python':platform.python_version(),
        'packages':{name:version(name) for name in ['torch','transformers','numpy','tokenizers','safetensors','mlir-aie','llvm-aie']},
        'bitsandbytes_source_commit':'c593d6f9debc17253c6b74fd6c2c1dbceecc4fcd',
        'bitsandbytes_source_version':bnb.__version__,
        'cpu_int8_backend':'torch._int_mm' if torch.backends.cpu.get_cpu_capability()=='AVX512' else 'FP32 matmul fallback',
        'model':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
        'host_dtype':'float32','offload_dtype':('int8/int32' if args.dq else 'bfloat16/float32') if pool else None,
        'dq_activation_quant_input_dtype':'float16' if args.dq else None,
        'dq_threshold':6.0 if args.dq else None,'attn_implementation':'eager',
        'batch_size':1,'input_tokens':128,'output_tokens':1,'prompt_name':'prompt_1',
        'prefix_reuse':False,'use_cache':True,'prefill':'one batched 128-token forward',
        'allocator_policy':'gc.collect and malloc_trim once after model preparation, before warmups; no trimming during timed inference',
        'replaced_modules':replaced,'npu_tensor_bytes':pool.tensor_bytes() if pool else 0,
        'npu_mappings':maps,'npu_mapped_kib':mapped_kib,
        'load_and_prepare_s':load_s,'memory_ready':ready,'memory_after':after,
        'lifetime_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'warmups':warmups,'runs':runs,'validations':validations}
    (out/'worker.json').write_text(json.dumps(report,indent=2)+'\n')
    emit(stage='done',label=args.label,validations=validations)


if __name__=='__main__':
    main()
