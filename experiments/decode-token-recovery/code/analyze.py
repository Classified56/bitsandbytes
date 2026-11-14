"""Recompute fidelity from saved timings, with prompt/block clustered intervals."""
import argparse,collections,json
from pathlib import Path
import numpy as np

def confusion(pred,truth):
    p,t=np.broadcast_arrays(np.asarray(pred,dtype=bool),np.asarray(truth,dtype=bool))
    return np.array([(p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum()],dtype=np.int64)

def rates(c):
    tp,fp,fn,tn=map(int,c);total=tp+fp+fn+tn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,fpr=fp/(fp+tn) if fp+tn else None,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,prevalence=(tp+fn)/total)

def eval_record(root,m,r):
    with np.load(root/r['file']) as z:
        ticks=z['ticks'];truth=z['truth'].astype(bool)
        assert ticks.shape==(r['repeats'],r['observed'],len(truth))
        hits=ticks<m['threshold'];consensus=(np.mean(np.mean(hits,axis=0)>=.6,axis=0)>=.5)
        filtered=consensus&~np.r_[False,consensus[:-1]]
        if 'columns' in z:
            expected=np.zeros(len(truth),dtype=bool);expected[z['columns']//32]=True;assert np.array_equal(expected,truth)
        first=z['order']==0
        first_counts=confusion(hits[first],np.broadcast_to(truth,hits.shape)[first])
        return dict(record=r,empty=not truth.any(),truth=truth,raw=confusion(hits,truth),consensus=confusion(consensus,truth),filtered=confusion(filtered,truth),first_probe=first_counts)

def aggregate(records):
    out=dict(windows=len(records),empty_windows=sum(r['empty'] for r in records))
    cluster='prompt' if 'prompt' in records[0]['record'] else 'block'
    keys=sorted({r['record'][cluster] for r in records});rng=np.random.default_rng(20261114)
    ix=rng.integers(len(keys),size=(5000,len(keys)))
    for stage in ['raw','consensus','filtered']:
        c=np.sum([r[stage] for r in records],axis=0);s=rates(c)
        sums=np.array([np.sum([r[stage] for r in records if r['record'][cluster]==k],axis=0) for k in keys])
        boot=sums[ix].sum(1)
        for metric,num,den in [('fpr',boot[:,1],boot[:,1]+boot[:,3]),('precision',boot[:,0],boot[:,0]+boot[:,1]),('recall',boot[:,0],boot[:,0]+boot[:,2])]:
            valid=den>0;s[metric+'_ci95']=np.quantile(num[valid]/den[valid],[.025,.975]).tolist() if valid.any() else None
        if stage!='raw':s['exact_windows']=int(sum(r[stage][1]==0 and r[stage][2]==0 for r in records))
        out[stage]=s
    out['first_probe_per_flush']=rates(np.sum([r['first_probe'] for r in records],axis=0))
    out['independent_clusters']=len(keys);return out

def analyze(root):
    m=json.loads((root/'manifest.json').read_text());assert m['complete'] and not m.get('pilot',False)
    evaluated=[eval_record(root,m,r) for r in m['records']];groups=collections.defaultdict(list)
    for r in evaluated:
        info=r['record'];fields=['phase','observed','condition'] if 'model' in m else ['kind','width','rows','observed','condition']
        groups[tuple(info[k] for k in fields)].append(r)
    rows=[dict(zip(fields,key),**aggregate(rs)) for key,rs in sorted(groups.items())]
    z=np.load(root/'calibration.npz');half=len(z['labels'])//2
    calibration=rates(confusion(z['ticks'][half:]<m['threshold'],z['labels'][half:]))
    if 'model' in m:
        assert len(m['validation'])==16 and all(v['bit_identical'] and v['baseline_sha256']==v['measured_sha256'] for v in m['validation'])
    # Matched first-row prefill/decode FPR contrasts, resampling whole prompts.
    contrasts=[]
    if 'model' in m:
        for condition in sorted({r['record']['condition'] for r in evaluated}):
            sub=[r for r in evaluated if r['record']['observed']==1 and r['record']['condition']==condition]
            diffs=[]
            for pid in range(16):
                pair=[]
                for phase in ['decode','prefill']:
                    chosen=[r for r in sub if r['record']['prompt']==pid and r['record']['phase']==phase]
                    if not chosen:break
                    pair.append(rates(np.sum([r['raw'] for r in chosen],axis=0))['fpr'])
                if len(pair)==2:diffs.append(pair[0]-pair[1])
            diffs=np.array(diffs);rng=np.random.default_rng(222);boot=diffs[rng.integers(len(diffs),size=(10000,len(diffs)))].mean(1)
            contrasts.append(dict(condition=condition,paired_prompts=len(diffs),decode_minus_prefill_fpr_pp=float(100*diffs.mean()),ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist()))
    report=dict(model=m.get('model'),rows=rows,threshold=m['threshold'],calibration=calibration,contrasts=contrasts,
        validation=dict(record_count=len(evaluated),bit_identical_forwards=sum(len(v['baseline_sha256']) for v in m.get('validation',[]))))
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(root,'PASS',len(evaluated),'records',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);a=p.parse_args()
    for root in a.folders:analyze(root)
