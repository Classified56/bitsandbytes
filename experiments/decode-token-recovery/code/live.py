"""Real prefill/decode gathers under the unmodified paper probe implementation."""
import argparse,time,types
from common import *
from probes import Engine,collect
import bitsandbytes.functional as F

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llama'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);cpus=setup(1);tok,model=load(a.model,a.checkpoint,a.profile)
    width=model.config.intermediate_size;e=Engine(capacity=192*width*2);threshold,labels,times=e.calibrate()
    np.savez_compressed(a.output/'calibration.npz',labels=labels,ticks=times,threshold=threshold)
    shared=torch.from_numpy(np.frombuffer(e.mapping,dtype=np.float16));modules=[(n,m) for n,m in model.named_modules() if isinstance(m,MixedInt8Linear) and n.endswith('down_proj')]
    manifest=dict(model=a.model,dtype='float16',threshold=threshold,cpus=cpus,pilot=a.pilot,records=[],validation=[],profile_sha256=digest(a.profile),protocol_sha256=digest(ROOT/'PROTOCOL.json'),start_unix=time.time())
    state=dict(active=False,prompt=0,step=0);conditions=PROTOCOL['live']['conditions'];start=time.monotonic()
    def wrapper(name,original):
        def forward(module,x):
            if not state['active']:return original(x)
            shape=x.shape;act=x.reshape(-1,shape[-1]);rows,k=act.shape
            assert act.dtype==torch.float16 and k%32==0
            ca,sca,cols=F.int8_vectorwise_quant(act,threshold=module.threshold)
            selected=np.array([] if cols is None else cols.numpy(),dtype=np.int64)
            assert np.array_equal(selected,np.flatnonzero((act.abs()>=module.threshold).any(0).numpy()))
            source=shared[:act.numel()].view_as(act);source.copy_(act)
            truth=np.zeros(k//32,dtype=np.uint8);truth[selected//32]=1;suba=[None]
            def gather():suba[0]=source[:,cols].contiguous() if len(selected) else None
            options=[(c,obs) for c in conditions for obs in ([1,4,16,32] if rows>1 else [1])]
            seed=10000+state['prompt']*1000+state['step']*100+int(name.split('.')[2]);random.Random(seed).shuffle(options)
            for condition,observed in options:
                ticks,order=collect(e,rows,k,2,gather,condition['schedule'],condition['repeats'],seed,observed)
                file=f'{len(manifest["records"]):06d}.npz'
                np.savez_compressed(a.output/file,ticks=ticks,order=order,truth=truth,columns=selected)
                manifest['records'].append(dict(file=file,prompt=state['prompt'],step=state['step'],phase='prefill' if rows>1 else 'decode',module=name,
                    rows=rows,observed=observed,condition=condition['name'],repeats=condition['repeats'],gathers=condition['gathers'],selected_columns=len(selected),activation_sha256=sha(act),outlier_threshold=module.threshold))
            if len(selected):
                assert torch.equal(suba[0],act[:,cols].contiguous())
                subb=torch.ops.bitsandbytes.int8_vectorwise_dequant.default(module.cb[:,cols].contiguous(),module.scb).to(act.dtype).t()
            product=torch.ops.bitsandbytes.int8_linear_matmul.default(ca,module.cb)
            y=torch.ops.bitsandbytes.int8_mm_dequant.default(product,sca,module.scb,dtype=act.dtype,bias=module.bias)
            if len(selected):y=y.addmm(suba[0],subb)
            return y.reshape(*shape[:-1],module.out_features)
        return forward
    for name,m in modules:m.forward=types.MethodType(wrapper(name,m.forward),m)
    def run(ids,active):
        state['active']=active;past=None;current=ids;hashes=[];generated=[]
        for step in range(1+(2 if a.pilot else 8)):
            if step and int(current) in stops(model):break
            state['step']=step;out=model(input_ids=current,past_key_values=past,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(out.logits).all();hashes.append(sha(out.logits));past=out.past_key_values
            current=out.logits[:,-1].argmax(-1).reshape(1,1);generated.append(int(current))
        state['active']=False;return hashes,generated
    for pid,topic in enumerate(topics()[152:152+(1 if a.pilot else 16)]):
        state['prompt']=pid;prompt,ids=inputs(tok,a.model,topic)
        b,bt=run(ids,False);r,rt=run(ids,True);assert b==r and bt==rt
        manifest['validation'].append(dict(prompt=pid,topic=topic,input_ids=ids[0].tolist(),generated_ids=rt,baseline_sha256=b,measured_sha256=r,bit_identical=True))
        save(a.output/'manifest.json',manifest)
        print(json.dumps(dict(model=a.model,prompt=pid,windows=len(manifest['records']),seconds=round(time.monotonic()-start,2))),flush=True)
    manifest.update(complete=True,seconds=time.monotonic()-start);save(a.output/'manifest.json',manifest);e.close()

if __name__=='__main__':main()
