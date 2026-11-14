import hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/dependencies'))
import torch
from benchmark_worker import AutoModelForCausalLM,AutoTokenizer
from linear_backends import MixedInt8Linear
PROTOCOL=json.loads((ROOT/'PROTOCOL.json').read_text())

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def setup(threads=1):
    chosen=[];cores=set()
    for c in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        key=((p/'physical_package_id').read_text(),(p/'core_id').read_text())
        if key not in cores:chosen.append(c);cores.add(key)
    os.sched_setaffinity(0,chosen[:threads]);torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(20261114)
    return chosen[:threads]

def topics():
    t=json.loads((ROOT/'topics.json').read_text());random.Random(PROTOCOL['topics_seed']).shuffle(t);return t

def inputs(tokenizer,key,topic):
    prompt=PROTOCOL['prompt'].format(topic=topic)
    ids=(tokenizer.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt') if key=='llama'
         else tokenizer(prompt+'\n\n',return_tensors='pt',add_special_tokens=False)['input_ids'])
    assert 32<=ids.shape[1]<=192
    return prompt,ids

def canonical(name):
    return name.replace('.k_proj','.q_proj').replace('.v_proj','.q_proj').replace('.up_proj','.gate_proj')

def load(key,checkpoint,profile=None):
    expected=PROTOCOL['models'][key]
    manifest=json.loads((ROOT/f'{key}-checkpoint-provenance.json').read_text())
    assert digest(checkpoint/'model.safetensors')==manifest['files']['model.safetensors']['sha256']
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float16,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
    assert (model.config.num_hidden_layers,model.config.hidden_size,model.config.intermediate_size)==(expected['layers'],expected['hidden'],expected['intermediate'])
    if profile:
        prof=json.loads(Path(profile).read_text());assert prof['model']==key and prof['quantile']==.999
        modules=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')]
        assert len(modules)==expected['layers']*7
        for n,m in modules:
            parent,child=n.rsplit('.',1)
            setattr(model.get_submodule(parent),child,MixedInt8Linear(m,pool=None,threshold=prof['thresholds'][canonical(n)]))
        del modules
    return tokenizer,model

def stops(model):
    e=model.generation_config.eos_token_id;return set(e if isinstance(e,list) else [e])

def sha(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()

def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
