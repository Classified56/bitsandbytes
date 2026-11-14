"""Frozen direct classifier on actual held-out cache masks, without retraining."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from classification import metric,network,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--direct',type=Path,required=True);p.add_argument('--classifiers',type=Path,required=True);p.add_argument('--live',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((a.direct/'config.json').read_text());manifest=json.loads((a.live/'manifest.json').read_text());frozen=json.loads((a.classifiers/'FROZEN.json').read_text())
    assert manifest['complete'] and manifest['profile_sha256']==cfg['profile_sha256'] and manifest['model']==cfg['model_key']
    ckpath=a.classifiers/'down32/classifier.pt';assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==frozen['down32']
    ck=torch.load(ckpath,weights_only=False);vary=ck['varying'];classes=ck['classes'];model=network(int(vary.sum()),len(classes));model.load_state_dict(ck['state_dict'])
    trainy=np.concatenate([np.load(f)['token_ids'] for f in sorted(a.direct.glob('*-train.npz'))]);modules=cfg['module_names']
    assert all(n.endswith('down_proj') for n in modules) and sum(cfg['dimensions'])//32==len(vary)
    val={r['prompt']:r for r in manifest['validation']};lookup={}
    for r in manifest['records']:
        if r['phase']=='decode':
            key=(r['condition'],r['prompt'],r['step'],r['module']);assert key not in lookup;lookup[key]=r
    conditions=sorted({k[0] for k in lookup});keys=sorted({(k[1],k[2]) for k in lookup});reports=[]
    y=np.array([val[pid]['generated_ids'][step-1] for pid,step in keys]);seq=np.array([pid for pid,step in keys]);direct_predictions=None
    for condition in conditions:
        xs={s:[] for s in ['direct','consensus','filtered']}
        for pid,step in keys:
            parts={s:[] for s in xs}
            for name in modules:
                r=lookup[condition,pid,step,name];z=np.load(a.live/r['file']);hit=z['ticks']<manifest['threshold'];truth=z['truth'].astype(bool)
                cons=(np.mean(np.mean(hit,0)>=.6,0)>=.5);filt=cons&~np.r_[False,cons[:-1]]
                for s,v in [('direct',truth),('consensus',cons),('filtered',filt)]:parts[s].append(v)
            for s in xs:xs[s].append(np.concatenate(parts[s]))
        preds={}
        for stage,x in xs.items():
            logits=predict(model,torch.from_numpy(np.stack(x)[:,vary].astype(np.float32)));preds[stage]=classes[np.argsort(-logits,axis=1)[:,:5]]
        if direct_predictions is None:direct_predictions=preds['direct']
        else:assert np.array_equal(direct_predictions,preds['direct'])
        for stage,pred in preds.items():
            r=metric(pred,y,seq,trainy);r.update(condition=condition,stage=stage)
            baseline=direct_predictions[:,0]==y;observed=pred[:,0]==y;responses=np.unique(seq)
            size=np.array([(seq==s).sum() for s in responses]);delta=np.array([(observed[seq==s].astype(int)-baseline[seq==s]).sum() for s in responses])
            ix=np.random.default_rng(1411).integers(len(responses),size=(10000,len(responses)));boot=delta[ix].sum(1)/size[ix].sum(1)
            r['observed_minus_direct_pp']=float(100*(observed.mean()-baseline.mean()));r['paired_gap_ci95_pp']=(100*np.quantile(boot,[.025,.975])).tolist();reports.append(r)
            np.savez_compressed(a.output/f'{condition}-{stage}.npz',token_ids=y,response_id=seq,position=np.array([s-1 for pid,s in keys]),predicted_ids=pred)
    out=dict(model=manifest['model'],profile_sha256=manifest['profile_sha256'],classifier_sha256=frozen['down32'],scope='predefined 16-topic cache cohort; direct-trained classifier transfer',results=reports)
    (a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(a.output,'PASS',len(y),'positions')

if __name__=='__main__':main()
