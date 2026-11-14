"""Direct token classification; train/dev and test access are separate stages."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[('down',g) for g in [1,4,16,32]]


def load(root,split,scope,g):
    c=json.loads((root/'config.json').read_text());files=sorted(root.glob(f'*-{split}.npz'))
    assert len(files)==c[split],(split,len(files),c[split])
    dims=c['dimensions'];ends=np.r_[0,np.cumsum(dims)];xs=[];ys=[];rs=[];ps=[]
    for f in files:
        z=np.load(f);ids=z['token_ids'];exact=np.unpackbits(z['exact'],axis=1)[:,:sum(dims)]
        assert exact.shape==(len(ids),sum(dims)) and 0<len(ids)<=c['steps']
        parts=[]
        for i,(name,d) in enumerate(zip(c['module_names'],dims)):
            if scope=='down' and not name.endswith('down_proj'):continue
            assert d%g==0
            parts.append(exact[:,ends[i]:ends[i+1]].reshape(len(ids),d//g,g).max(2))
        xs.append(np.concatenate(parts,axis=1));ys.append(ids)
        rs.extend([int(f.stem.split('-')[0])]*len(ids));ps.extend(range(len(ids)))
    return np.concatenate(xs),np.concatenate(ys),np.array(rs),np.array(ps)


def network(n,k):return nn.Sequential(nn.Linear(n,1024),nn.ReLU(),nn.Linear(1024,k))

def predict(model,x):
    model.eval()
    with torch.no_grad():return torch.cat([model(t) for t in x.split(256)]).numpy()


def train(data,output,scope,g):
    output.mkdir(parents=True,exist_ok=False);torch.manual_seed(0)
    x,y,_,_=load(data,'train',scope,g);v,vy,_,_=load(data,'dev',scope,g)
    classes=np.unique(y);vary=x.var(0)>1e-7
    assert vary.any(),'No varying direct-channel features; record as uninformative, do not change threshold against test'
    xt=torch.from_numpy(x[:,vary].astype(np.float32));vt=torch.from_numpy(v[:,vary].astype(np.float32));del x,v
    target=torch.from_numpy(np.searchsorted(classes,y)).long();model=network(int(vary.sum()),len(classes))
    opt=torch.optim.AdamW(model.parameters(),lr=.001);best=-1;history=[];start=time.monotonic()
    for epoch in range(1,41):
        model.train();losses=[]
        for ix in torch.randperm(len(y)).split(128):
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(xt[ix]),target[ix]);loss.backward();opt.step();losses.append(float(loss.detach()))
        logits=predict(model,vt);score=float((classes[logits.argmax(1)]==vy).mean())
        row=dict(epoch=epoch,loss=float(np.mean(losses)),dev_top1=score,seconds=time.monotonic()-start);history.append(row)
        if score>best:
            best=score
            torch.save(dict(state_dict=model.state_dict(),classes=classes,varying=vary,scope=scope,group_width=g,best_epoch=epoch,dev_top1=score),output/'classifier.pt')
        print(json.dumps(dict(scope=scope,group_width=g,**row)),flush=True)
    provenance=dict(scope=scope,group_width=g,training_positions=len(y),development_positions=len(vy),training_classes=len(classes),
        input_features=len(vary),varying_features=int(vary.sum()),history=history,checkpoint_sha256=hashlib.sha256((output/'classifier.pt').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_read=False)
    (output/'training.json').write_text(json.dumps(provenance,indent=2)+'\n')


def metric(pred5,truth,seq,trainlabels):
    pred=pred5[:,0];correct=pred==truth;classes,counts=np.unique(trainlabels,return_counts=True)
    known=np.isin(truth,classes);top20=classes[np.argsort(-counts,kind='stable')[:20]];outside=~np.isin(truth,top20)
    responses=np.unique(seq);sizes=np.array([(seq==s).sum() for s in responses]);hits=np.array([correct[seq==s].sum() for s in responses])
    rng=np.random.default_rng(911);ix=rng.integers(len(responses),size=(10000,len(responses)))
    ci=np.quantile(hits[ix].sum(1)/sizes[ix].sum(1),[.025,.975])
    return dict(positions=len(truth),correct=int(correct.sum()),top1=float(correct.mean()),top5=float((pred5==truth[:,None]).any(1).mean()),
        balanced_accuracy=float(np.mean([correct[truth==t].mean() for t in np.unique(truth)])),test_token_types=len(np.unique(truth)),
        known_fraction=float(known.mean()),known_top1=float(correct[known].mean()) if known.any() else None,
        unseen_positions=int((~known).sum()),unseen_correct=int(correct[~known].sum()),
        exact_response_prefixes=int(sum(bool(correct[seq==s].all()) for s in responses)),responses=len(responses),
        ci95_response_bootstrap=ci.tolist(),outside_training_top20_positions=int(outside.sum()),
        outside_training_top20_top1=float(correct[outside].mean()) if outside.any() else None,
        correct_from_training_top20=int((correct&~outside).sum()))


def evaluate(data,out):
    frozen=json.loads((out/'FROZEN.json').read_text());reports={}
    trainfiles=sorted(data.glob('*-train.npz'))
    train_sequences=[]
    for f in trainfiles:
        with np.load(f) as z:train_sequences.append(z['token_ids'].copy())
    trainy=np.concatenate(train_sequences)
    for scope,g in VARIANTS:
        name=f'{scope}{g}';folder=out/name;ckpath=folder/'classifier.pt'
        assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==frozen[name]
        ck=torch.load(ckpath,weights_only=False);classes=ck['classes'];vary=ck['varying']
        x,y,seq,position=load(data,'test',scope,g)
        model=network(int(vary.sum()),len(classes));model.load_state_dict(ck['state_dict'])
        logits=predict(model,torch.from_numpy(x[:,vary].astype(np.float32)));del x
        top=classes[np.argsort(-logits,axis=1)[:,:5]]
        report=metric(top,y,seq,trainy);report.update(scope=scope,group_width=g,best_epoch=ck['best_epoch'],dev_top1=ck['dev_top1'])
        shuffled=top[np.random.default_rng(121).permutation(len(top))]
        report['shuffled_test_features']=metric(shuffled,y,seq,trainy)
        np.savez_compressed(folder/'test-predictions.npz',token_ids=y,response_id=seq,position=position,predicted_ids=top)
        (folder/'test-results.json').write_text(json.dumps(report,indent=2)+'\n');reports[name]=report
    u,c=np.unique(trainy,return_counts=True);freq=u[np.argsort(-c,kind='stable')[:5]]
    freqt=np.tile(freq,(len(y),1));position_top5={}
    for pos in np.unique(position):
        vals=np.array([tokens[pos] for tokens in train_sequences if len(tokens)>pos])
        pu,pc=np.unique(vals,return_counts=True);choices=pu[np.argsort(-pc,kind='stable')[:5]]
        position_top5[pos]=np.pad(choices,(0,5-len(choices)),mode='edge')
    pospred=[position_top5[pos] for pos in position]
    baselines=dict(frequency=metric(freqt,y,seq,trainy),position=metric(np.stack(pospred),y,seq,trainy))
    records=[json.loads(s) for s in (data/'responses.jsonl').read_text().splitlines()]
    assert len(records)==sum(json.loads((data/'config.json').read_text())[s] for s in ['train','dev','test'])
    hashes={s:{r['prompt_sha256'] for r in records if r['split']==s} for s in ['train','dev','test']}
    assert not(hashes['train']&hashes['test'] or hashes['train']&hashes['dev'] or hashes['dev']&hashes['test'])
    texts={s:{tuple(r['observed_ids']) for r in records if r['split']==s} for s in hashes}
    overlap=len(texts['train']&texts['test'])
    assert overlap==0,'Repeated response prefix across train/test: report and investigate, do not silently remove'
    results=dict(model=json.loads((data/'config.json').read_text())['model'],feature_source='direct operator-selected columns; no side channel',
        training_token_types=len(np.unique(trainy)),results=reports,baselines=baselines,
        validation=dict(prompt_split_disjoint=True,identical_train_test_response_prefixes=overlap,recording_checks=[r['validation'] for r in records if r['validation']]))
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=[results['model']+': DIRECT-CHANNEL DECODE TOKEN CLASSIFICATION','',
           'Features: actual DQ-selected columns, frozen per-layer P99.9 thresholds; test OOV tokens count as errors.',
           'Scope   Group   Correct/N      Acc.%   Balanced Acc.%   Top5%   Exact prefixes']
    for name,r in reports.items():
        lines.append(f'{r["scope"]:<7} {r["group_width"]:>5}   {r["correct"]:>5}/{r["positions"]:<5} {100*r["top1"]:>8.2f} {100*r["balanced_accuracy"]:>16.2f} {100*r["top5"]:>8.2f} {r["exact_response_prefixes"]:>6}/{r["responses"]}')
    lines+=['','Baseline       Acc.%    Top5%']
    for name,r in baselines.items():lines.append(f'{name:<13} {100*r["top1"]:>8.2f} {100*r["top5"]:>8.2f}')
    lines+=['',f'Training token types: {results["training_token_types"]}; test coverage: {100*reports["down1"]["known_fraction"]:.2f}%.',
            'Direct classifier only; no cache noise, LM fusion, or one-shot side-channel claim.']
    (out/'FINAL_RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='validation'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['train','evaluate'],required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (a.data/'COMPLETE.json').exists(),'Refusing partial acquisition'
    if a.stage=='train':
        a.output.mkdir(parents=True,exist_ok=False)
        for scope,g in VARIANTS:train(a.data,a.output/f'{scope}{g}',scope,g)
        (a.output/'FROZEN.json').write_text(json.dumps({f'{s}{g}':hashlib.sha256((a.output/f'{s}{g}/classifier.pt').read_bytes()).hexdigest() for s,g in VARIANTS},indent=2)+'\n')
    else:evaluate(a.data,a.output)

if __name__=='__main__':main()
