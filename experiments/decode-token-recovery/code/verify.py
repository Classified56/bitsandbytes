"""Independent counts, label alignment, and cross-platform artifact checks."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from prefill_flush_reload.scoring import candidate_runs,score_groups

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def check_predictions(p,r):
    z=np.load(p);y=z['token_ids'];pred=z['predicted_ids'];seq=z['response_id'];hit=pred[:,0]==y
    assert len(y)==r['positions'] and int(hit.sum())==r['correct']
    assert np.isclose(hit.mean(),r['top1']) and np.isclose((pred==y[:,None]).any(1).mean(),r['top5'])
    assert np.isclose(np.mean([hit[y==t].mean() for t in np.unique(y)]),r['balanced_accuracy'])
    assert sum(bool(hit[seq==s].all()) for s in np.unique(seq))==r['exact_response_prefixes']
    return z

def main():
    report=dict(checks='PASS',direct={},cross_platform={},bridge={})
    provenance=read(ROOT/'vendor-provenance.json')
    for path,expected in provenance['unmodified_files'].items():assert sha(ROOT/'vendor'/path)==expected,path
    environments={p:read(ROOT/f'environment-{p}.json') for p in ['intel','amd']}
    assert environments['intel']['backend_sha256']==environments['amd']['backend_sha256']
    for path,expected in environments['intel']['backend_sha256'].items():assert sha(ROOT/'runtime/vendor/bitsandbytes'/path)==expected,path
    report['provenance']=dict(unmodified_probe_files=len(provenance['unmodified_files']),identical_backend_files=len(environments['intel']['backend_sha256']))
    report['upstream_scoring']={}
    for platform in ['intel','amd']:
        for dataset in ['controls','live-qwen','live-llama']:
            folder=ROOT/f'results/{platform}/{dataset}';manifest=read(folder/'manifest.json');analysis=read(folder/'analysis.json')
            fields=['phase','observed','condition'] if 'model' in manifest else ['kind','width','rows','observed','condition']
            expected={tuple(r[k] for k in fields):r for r in analysis['rows']};sums=collections.defaultdict(lambda:np.zeros((2,4),dtype=np.int64))
            for rec in manifest['records']:
                with np.load(folder/rec['file']) as z:
                    hits=z['ticks']<manifest['threshold'];truth=z['truth'].astype(bool)
                vote=np.mean(np.mean(hits,axis=0)>=.6,axis=0)>=.5;candidates=(32*np.flatnonzero(vote)).tolist()
                guessed=[r['guessed_group'] for r in candidate_runs(candidates)]
                actual=(32*np.flatnonzero(truth)).tolist();universe=(32*np.arange(len(truth))).tolist();key=tuple(rec[k] for k in fields)
                for i,pred in enumerate([candidates,guessed]):
                    score=score_groups(pred,actual,universe);sums[key][i]+=[score[n] for n in ['tp','fp','fn','tn']]
            for key,count in sums.items():
                for i,stage in enumerate(['consensus','filtered']):assert count[i].tolist()==[expected[key][stage][n] for n in ['tp','fp','fn','tn']]
            report['upstream_scoring'][f'{platform}-{dataset}']=dict(records=len(manifest['records']),exact_match=True)
    for key in ['qwen','llama']:
        d=ROOT/f'results/amd/direct-{key}';c=ROOT/f'results/amd/classifiers-{key}';cfg=read(d/'config.json');res=read(c/'results.json');frozen=read(c/'FROZEN.json')
        assert (d/'COMPLETE.json').exists() and not cfg['pilot'] and cfg['profile_sha256']==sha(ROOT/f'profiles/{key}.json')
        records=[json.loads(s) for s in (d/'responses.jsonl').read_text().splitlines()];assert len(records)==200
        byid={r['id']:r for r in records};ends=np.r_[0,np.cumsum(cfg['dimensions'])];summary={}
        for split,n in [('train',128),('dev',24),('test',48)]:
            files=sorted(d.glob(f'*-{split}.npz'));assert len(files)==n;total=empty=0;columnsum=0
            for f in files:
                z=np.load(f);ids=z['token_ids'];x=np.unpackbits(z['exact'],axis=1)[:,:ends[-1]]
                counts=np.stack([x[:,ends[i]:ends[i+1]].sum(1) for i in range(len(cfg['dimensions']))],1)
                assert np.array_equal(counts,z['counts']);record=byid[int(f.stem.split('-')[0])]
                assert ids.tolist()==record['observed_ids']==record['generated_ids'][:len(ids)]
                assert not set(ids.tolist())&set(cfg['stopping_token_ids']);assert 0<len(ids)<=32
                total+=len(ids);empty+=int((counts.sum(1)==0).sum());columnsum+=int(counts.sum())
            summary[split]=dict(responses=n,positions=total,empty_positions=empty,mean_columns=columnsum/total)
        for name,r in res['results'].items():
            assert sha(c/name/'classifier.pt')==frozen[name];z=check_predictions(c/name/'test-predictions.npz',r)
            for tid,s,pos in zip(z['token_ids'],z['response_id'],z['position']):assert byid[int(s)]['observed_ids'][int(pos)]==tid
        for rec in read(ROOT/f'profiles/{key}.json')['calibration_responses']:
            assert rec['id']<128 and rec['input_ids']==byid[rec['id']]['input_ids']
        report['direct'][key]=summary
        manifests={platform:read(ROOT/f'results/{platform}/live-{key}/manifest.json') for platform in ['intel','amd']}
        for m in manifests.values():assert m['complete'] and m['profile_sha256']==cfg['profile_sha256']
        vi,va=manifests['intel']['validation'],manifests['amd']['validation'];assert len(vi)==len(va)==16
        matching=0;compared=0
        for x,y in zip(vi,va):
            assert x['topic']==y['topic'] and x['input_ids']==y['input_ids']
            assert x['bit_identical'] and y['bit_identical']
            compared+=min(len(x['generated_ids']),len(y['generated_ids']))
            matching+=sum(a==b for a,b in zip(x['generated_ids'],y['generated_ids']))
        ri={(r['prompt'],r['step'],r['module']):r for r in manifests['intel']['records'] if r['condition']=='single_execution' and r['observed']==1}
        ra={(r['prompt'],r['step'],r['module']):r for r in manifests['amd']['records'] if r['condition']=='single_execution' and r['observed']==1}
        keys=sorted(set(ri)&set(ra));colmatch=hashmatch=0
        for k in keys:
            x=np.load(ROOT/f'results/intel/live-{key}'/ri[k]['file']);y=np.load(ROOT/f'results/amd/live-{key}'/ra[k]['file'])
            colmatch+=bool(np.array_equal(x['columns'],y['columns']));hashmatch+=ri[k]['activation_sha256']==ra[k]['activation_sha256']
        report['cross_platform'][key]=dict(identical_input_sequences=True,matching_generated_ids=matching,compared_generated_ids=compared,
            matching_column_sets=colmatch,compared_windows=len(keys),bit_identical_activation_windows=hashmatch,
            interpretation='Live CPU numerical paths may differ; fixed-column synthetic controls isolate the platform with identical truth.')
        for platform in ['intel','amd']:
            b=ROOT/f'results/{platform}/bridge-{key}';br=read(b/'results.json');assert br['classifier_sha256']==frozen['down32']
            val={r['prompt']:r for r in manifests[platform]['validation']}
            for r in br['results']:
                z=check_predictions(b/f'{r["condition"]}-{r["stage"]}.npz',r)
                for token,prompt,pos in zip(z['token_ids'],z['response_id'],z['position']):
                    assert int(token)==val[int(prompt)]['generated_ids'][int(pos)]
            report['bridge'][f'{platform}-{key}']=dict(conditions=len(br['results']),positions=br['results'][0]['positions'])
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
