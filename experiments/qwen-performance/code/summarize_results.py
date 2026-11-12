"""Audit all six raw measurements and generate performance-only final tables."""
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LABELS=[('intel_cpu_no_dq','Intel CPU, without DQ'),
        ('intel_cpu_dq','Intel CPU, with DQ'),
        ('amd_cpu_no_dq','AMD CPU, without DQ'),
        ('amd_cpu_dq','AMD CPU, with DQ'),
        ('amd_npu_no_dq','AMD CPU+NPU, without DQ'),
        ('amd_npu_dq','AMD CPU+NPU, with DQ')]
workers={}
summaries={}
vectors={}
for key,_ in LABELS:
    folder=ROOT/'results'/key
    raw=json.loads((folder/'worker.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    assert raw['packages']['torch']=='2.11.0+cpu'
    assert raw['packages']['transformers']=='4.57.1'
    assert raw['packages']['numpy']=='2.2.6'
    assert raw['input_tokens']==128 and raw['batch_size']==1 and raw['output_tokens']==1
    assert raw['host_dtype']=='float32' and raw['torch_threads']==4
    assert not raw['prefix_reuse'] and 'malloc_trim' in raw['allocator_policy']
    assert len(raw['runs'])==10 and len(raw['warmups'])==2
    assert abs(statistics.median(x['ttft_ms'] for x in raw['runs'])-summary['ttft_median_ms'])<1e-9
    assert len(set(x['first_token'] for x in raw['runs']))==1
    for run in raw['runs']:
        assert run['memory_after']['VmSwap_kib']==0
        assert run['npu_dispatches']==(168 if raw['configuration']['backend']=='npu' else 0)
        assert bool(run['outlier_columns_sum'])==bool(raw['configuration']['dq'])
    markers={}
    for line in (folder/'worker.log').read_text().splitlines():
        try:
            event=json.loads(line)
            if event.get('stage') in ('measured_start','measured_end'):markers[event['stage']]=event
        except json.JSONDecodeError:pass
    samples=json.loads((folder/'rss-samples.json').read_text())
    selected=[s for s in samples if markers['measured_start']['monotonic_ns']<=s['monotonic_ns']<=markers['measured_end']['monotonic_ns']]
    assert max(s['VmRSS_kib'] for s in selected)/1024==summary['host_peak_rss_mib']
    assert all(m['Rss_kib']==0 for m in raw['npu_mappings'])
    assert sum(m['Size_kib'] for m in raw['npu_mappings'])/1024==summary['npu_mapped_allocation_mib']
    assert summary['host_peak_rss_mib']+summary['npu_mapped_allocation_mib']==summary['peak_accounted_memory_mib']
    workers[key],summaries[key]=raw,summary
    vectors[key]=dict(np.load(folder/'validation-logits.npz'))

pairs=[('intel_cpu_no_dq','amd_cpu_no_dq'),('intel_cpu_dq','amd_cpu_dq'),
       ('amd_cpu_no_dq','amd_npu_no_dq'),('amd_cpu_dq','amd_npu_dq'),
       ('intel_cpu_no_dq','intel_cpu_dq'),('amd_cpu_no_dq','amd_cpu_dq')]
checks=[]
for left,right in pairs:
    for prompt in vectors[left]:
        a,b=vectors[left][prompt].astype(np.float64),vectors[right][prompt].astype(np.float64)
        cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        assert cosine>0.995, (left,right,prompt,cosine)
        item={'left':left,'right':right,'prompt':prompt,'cosine_similarity':cosine,
              'rmse':float(np.sqrt(np.mean((a-b)**2))),
              'max_abs_difference':float(np.max(np.abs(a-b))),
              'left_token':int(np.argmax(a)),'right_token':int(np.argmax(b)),
              'top1_match':int(np.argmax(a))==int(np.argmax(b)),
              'exact_logits_match':bool(np.array_equal(a,b))}
        checks.append(item)
(ROOT/'results/cross-validation.json').write_text(json.dumps({'checks':checks},indent=2)+'\n')
(ROOT/'results/comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
lines=['Qwen2.5-0.5B | 128 input tokens | batch 1 | first output token | 4 CPU threads',
       '2 warmups + 10 measured requests/configuration | median TTFT | no prefix reuse',
       '',
       '| Configuration                  | TTFT median (ms) | TTFT min-max (ms) | Peak memory (MiB) |',
       '|--------------------------------|------------------|-------------------|-------------------|']
for key,name in LABELS:
    s=summaries[key]
    interval=f"{s['ttft_min_ms']:.2f}-{s['ttft_max_ms']:.2f}"
    lines.append(f"| {name:<30} | {s['ttft_median_ms']:>16.2f} | {interval:>17} | {s['peak_accounted_memory_mib']:>17.2f} |")
lines+=['',
        '| Configuration                  | Peak host RSS (MiB) | NPU mapped allocation (MiB) |',
        '|--------------------------------|---------------------|-----------------------------|']
for key,name in LABELS:
    s=summaries[key]
    lines.append(f"| {name:<30} | {s['host_peak_rss_mib']:>19.2f} | {s['npu_mapped_allocation_mib']:>27.2f} |")
lines+=['',
        'Peak memory = inference-phase host RSS + mapped NPU buffer allocation (shared DDR, excluded from RSS).',
        'NPU mapping size is allocated/mapped memory; physical residency was not separately measured.',
        'Loading/JIT excluded. CPU FP32; non-DQ NPU matmuls BF16; DQ threshold 6.0, INT8 + FP32 outlier correction.']
(ROOT/'FINAL_RESULTS.txt').write_text('\n'.join(lines)+'\n')
paper=ROOT/'paper'
paper.mkdir(exist_ok=True)
rows=[]
for key,name in LABELS:
    s=summaries[key]
    rows.append(f"        {name:<30} & {s['ttft_median_ms']:.2f} & {s['peak_accounted_memory_mib']:.2f} \\\\")
table=r'''\begin{table}[t]
    \centering
    \caption{Qwen2.5-0.5B inference performance. DQ denotes dynamic
    quantization with outlier correction. TTFT is the median of 10 warm
    requests with 128 input tokens, batch size 1, and four CPU threads;
    model loading and compilation are excluded.}
    \label{tab:architecture-performance}
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}lrr@{}}
        \hline
        Execution configuration & TTFT (ms) & Peak memory (MiB) \\
        \hline
'''+ '\n'.join(rows)+r'''
        Rocket+Gemmini, with DQ & TBD & TBD \\
        \hline
    \end{tabular}
    \vspace{2pt}
    \parbox{\columnwidth}{\scriptsize Peak memory is sampled inference-phase
    host RSS plus mapped NPU buffer allocation in shared DDR; the latter is
    excluded from RSS and is not a separate physical-residency measurement.
    CPU baselines use FP32. Single-core NPU matmuls use BF16 without DQ; DQ uses INT8 with
    FP32 CPU outlier correction. Rocket+Gemmini is not measured here.}
\end{table}
'''
(paper/'table.tex').write_text(table)
print('\n'.join(lines))
print('Validated',len(checks),'cross-configuration prompt comparisons.')
