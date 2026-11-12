"""External own-child RSS sampler, independent of the model worker's GIL."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


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
    command=[sys.executable,'-u',str(ROOT/'code/benchmark_worker.py'),
             '--label',args.label,'--backend',args.backend,'--dq',str(args.dq),
             '--runs',str(args.runs),'--warmups',str(args.warmups)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    samples=[]
    stop=threading.Event()
    phases={}
    def monitor():
        while not stop.is_set():
            try:
                text=Path(f'/proc/{process.pid}/status').read_text()
                sample={'monotonic_ns':time.monotonic_ns()}
                for line in text.splitlines():
                    key,_,value=line.partition(':')
                    if key in ('VmRSS','VmHWM','VmSwap'):
                        sample[key+'_kib']=int(value.split()[0])
                if 'VmRSS_kib' in sample:samples.append(sample)
            except (FileNotFoundError,ProcessLookupError):break
            stop.wait(0.005)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    with (out/'worker.log').open('w') as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line,end='',flush=True)
            try:
                event=json.loads(line)
                if event.get('stage') in ('measured_start','measured_end'):
                    phases[event['stage']]=event
            except json.JSONDecodeError:pass
    code=process.wait()
    stop.set()
    thread.join()
    (out/'rss-samples.json').write_text(json.dumps(samples)+'\n')
    assert code==0, f'worker failed with exit code {code}'
    selected=[s for s in samples if phases['measured_start']['monotonic_ns']<=s['monotonic_ns']<=phases['measured_end']['monotonic_ns']]
    assert selected
    raw=json.loads((out/'worker.json').read_text())
    ttft=[r['ttft_ms'] for r in raw['runs']]
    assert len(ttft)==args.runs
    host_peak=max(s['VmRSS_kib'] for s in selected)
    npu_kib=raw['npu_mapped_kib']
    gaps=[(b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(selected,selected[1:])]
    summary={'label':args.label,'runs':len(ttft),'ttft_median_ms':statistics.median(ttft),
        'ttft_mean_ms':statistics.mean(ttft),'ttft_min_ms':min(ttft),'ttft_max_ms':max(ttft),
        'host_peak_rss_mib':host_peak/1024,'npu_mapped_allocation_mib':npu_kib/1024,
        'peak_accounted_memory_mib':(host_peak+npu_kib)/1024,
        'npu_tensor_allocation_mib':raw['npu_tensor_bytes']/1048576,
        'rss_sample_count':len(selected),'sampling_mean_interval_ms':statistics.mean(gaps) if gaps else None,
        'sampling_max_interval_ms':max(gaps) if gaps else None,
        'memory_definition':'Peak sampled host RSS plus mapped XRT BO allocation (shared DDR, zero RSS); warmed inference phase only. NPU mapping size is allocated/mapped memory, not separately measured physical residency.',
        'ttft_definition':'Raw-text tokenization + fresh 128-token batched model forward + greedy first-token selection and text decode; excludes load, JIT, warmups, network, queueing.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'stage':'summary',**summary}),flush=True)


if __name__=='__main__':main()
