"""Matched no-gather and fixed-column shape controls on owned buffers."""
import argparse,ctypes as ct,itertools,time
from common import *
from probes import Engine,collect

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cpus=setup(1)
    e=Engine(capacity=64*8192*2);cut,labels,ticks=e.calibrate()
    np.savez_compressed(a.output/'calibration.npz',labels=labels,ticks=ticks,threshold=cut)
    cols=np.array(PROTOCOL['controls']['selected_columns'],dtype=np.uint64);nativecols=(ct.c_size_t*len(cols))(*cols);dst=np.zeros((64,len(cols)),dtype=np.float16)
    conditions=PROTOCOL['live']['conditions'];jobs=[]
    for block,width,(rows,observed),c in itertools.product(range(10),[4864,8192],[(1,1),(64,1),(64,4),(64,16),(64,32)],conditions):
        for kind in ['no_gather','fixed_columns']:jobs.append((block,width,rows,observed,c,kind))
    for block,width in itertools.product(range(10),[4864,8192]):jobs.append((block,width,1,1,dict(name='fresh_target',schedule='single',repeats=1,gathers=0),'no_gather'))
    np.random.default_rng(942).shuffle(jobs);manifest=dict(threshold=cut,cpus=cpus,records=[],protocol_sha256=digest(ROOT/'PROTOCOL.json'));start=time.monotonic()
    for i,(block,width,rows,observed,c,kind) in enumerate(jobs):
        values=np.frombuffer(e.mapping,dtype=np.float16,count=rows*width).reshape(rows,width);values[:]=1
        if kind=='fixed_columns':
            for j,col in enumerate(cols):values[(j*17)%rows,col]=48 if j%2==0 else -48
        def gather():
            if kind=='fixed_columns':e.native.gather(e.address,dst.ctypes.data,nativecols,len(cols),rows,width)
        t,order=collect(e,rows,width,2,gather,c['schedule'],c['repeats'],20261114+block,observed)
        truth=np.zeros(width//32,dtype=np.uint8)
        if kind=='fixed_columns':
            truth[cols//32]=1;assert np.array_equal(dst.reshape(-1)[:rows*len(cols)].reshape(rows,len(cols)),values[:,cols])
        f=f'{i:05d}.npz';np.savez_compressed(a.output/f,ticks=t,order=order,truth=truth)
        manifest['records'].append(dict(file=f,block=block,width=width,rows=rows,observed=observed,condition=c['name'],repeats=c['repeats'],gathers=c['gathers'] if kind=='fixed_columns' else 0,kind=kind))
        if i%50==49:print('controls',i+1,'/',len(jobs),round(time.monotonic()-start,2),flush=True)
    manifest.update(complete=True,seconds=time.monotonic()-start);save(a.output/'manifest.json',manifest);e.close()

if __name__=='__main__':main()
