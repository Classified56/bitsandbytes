"""Capture direct decode labels and actual profiled-DQ selections."""
import argparse,time,fcntl
from common import *
import bitsandbytes.functional as F

class Recorder:
    def __init__(self,model):
        self.modules=[(n,m) for n,m in model.named_modules() if isinstance(m,MixedInt8Linear) and n.endswith('down_proj')]
        self.active=None;self.enabled=False;self.masks=[];self.original=F.int8_vectorwise_quant
        for i,(_,m) in enumerate(self.modules):
            m.register_forward_pre_hook(self.pre(i));m.register_forward_hook(self.post)
        F.int8_vectorwise_quant=self.quant
    def pre(self,i):
        def hook(module,args):self.active=i
        return hook
    def post(self,*args):self.active=None
    def quant(self,a,*args,**kwargs):
        out=self.original(a,*args,**kwargs)
        if self.enabled and self.active is not None:
            assert len(self.masks)==self.active and a.shape[0]==1
            mask=np.zeros(a.shape[1],dtype=np.uint8)
            if out[2] is not None:mask[out[2].numpy()]=1
            assert np.array_equal(mask,(a.abs()>=self.modules[self.active][1].threshold).numpy()[0])
            self.masks.append(mask)
        return out

@torch.inference_mode()
def run(model,rec,ids,steps,record):
    rec.enabled=False;out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
    pre=sha(out.logits);past=out.past_key_values;token=out.logits[:,-1].argmax(-1).reshape(1,1)
    consumed=[];generated=[int(token)];packed=[];counts=[];hashes=[]
    for _ in range(steps):
        if int(token) in stops(model):break
        consumed.append(int(token));rec.enabled=record;rec.masks=[]
        out=model(input_ids=token,past_key_values=past,use_cache=True,logits_to_keep=1)
        if record:
            assert len(rec.masks)==len(rec.modules)
            packed.append(np.packbits(np.concatenate(rec.masks)));counts.append([int(x.sum()) for x in rec.masks])
        assert torch.isfinite(out.logits).all();hashes.append(sha(out.logits));past=out.past_key_values
        token=out.logits[:,-1].argmax(-1).reshape(1,1);generated.append(int(token))
    rec.enabled=False
    assert consumed and consumed==generated[:len(consumed)]
    return dict(consumed=consumed,generated=generated,exact=packed,counts=counts,hashes=hashes,prefill_hash=pre)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llama'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/('.'+a.output.name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (a.output/'COMPLETE.json').exists():
        existing=json.loads((a.output/'config.json').read_text())
        assert existing['model_key']==a.model and existing['profile_sha256']==digest(a.profile) and existing['protocol_sha256']==digest(ROOT/'PROTOCOL.json') and existing['pilot']==a.pilot
        for split in ['train','dev','test']:assert len(list(a.output.glob(f'*-{split}.npz')))==existing[split]
        print('Reusing verified completed acquisition',a.output,flush=True);return
    a.output.mkdir(parents=True,exist_ok=False);cpus=setup(4);tok,model=load(a.model,a.checkpoint,a.profile);rec=Recorder(model)
    sizes=(2,1,1) if a.pilot else (128,24,48);steps=4 if a.pilot else 32
    cfg=dict(model=PROTOCOL['models'][a.model]['repo'],model_key=a.model,train=sizes[0],dev=sizes[1],test=sizes[2],steps=steps,pilot=a.pilot,
        cpus=cpus,module_names=[n for n,m in rec.modules],dimensions=[m.in_features for n,m in rec.modules],stopping_token_ids=sorted(stops(model)),
        profile_sha256=digest(a.profile),protocol_sha256=digest(ROOT/'PROTOCOL.json'),dtype='float16',feature_source='actual operator-selected columns',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').rglob('*.py')})
    save(a.output/'config.json',cfg);start=time.monotonic()
    for i,topic in enumerate(topics()[:sum(sizes)]):
        split='train' if i<sizes[0] else 'dev' if i<sum(sizes[:2]) else 'test';prompt,ids=inputs(tok,a.model,topic)
        r=run(model,rec,ids,steps,True);validation=None
        if i<2:
            b=run(model,rec,ids,steps,False)
            assert r['hashes']==b['hashes'] and r['prefill_hash']==b['prefill_hash'] and r['generated']==b['generated']
            validation=dict(bit_identical=True,baseline_logit_sha256=b['hashes'])
        np.savez_compressed(a.output/f'{i:04d}-{split}.npz',token_ids=np.array(r['consumed'],dtype=np.int32),exact=np.stack(r['exact']),counts=np.array(r['counts'],dtype=np.uint16))
        record=dict(id=i,split=split,topic=topic,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids=ids[0].tolist(),
            observed_ids=r['consumed'],generated_ids=r['generated'],observed_text=tok.decode(r['consumed']),logit_sha256=r['hashes'],validation=validation)
        with (a.output/'responses.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(id=i,split=split,positions=len(r['consumed']),selected_columns=float(np.array(r['counts']).sum(1).mean()),seconds=round(time.monotonic()-start,2))),flush=True)
    save(a.output/'COMPLETE.json',dict(responses=sum(sizes),seconds=time.monotonic()-start))

if __name__=='__main__':main()
