"""Concise tables from completed and independently verified artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def fmt(x):return '--' if x is None else f'{100*x:.2f}'

def main():
    assert read(ROOT/'verification.json')['checks']=='PASS'
    combined={};lines=['DECODE EXTENSION: RECONSTRUCTED P99.9 BASELINE','FP16; training-only frozen per-layer thresholds; reconstruction assumptions in PROTOCOL.json.','',
        'V-A: DIRECT DECODE TOKEN CLASSIFICATION',
        'Model  Group  Correct/N     Top1%  Balanced%  Top5%  Exact prefixes']
    va=[];vb=[];bridge=[]
    for key in ['qwen','llama']:
        r=read(ROOT/f'results/amd/classifiers-{key}/results.json');combined[f'direct-{key}']=r
        for name,m in r['results'].items():
            lines.append(f'{key:<6} {m["group_width"]:>5} {m["correct"]:>5}/{m["positions"]:<5} {fmt(m["top1"]):>7} {fmt(m["balanced_accuracy"]):>10} {fmt(m["top5"]):>6} {m["exact_response_prefixes"]:>7}/{m["responses"]}')
            va.append(f'{key.capitalize()} & {m["group_width"]} & {m["correct"]}/{m["positions"]} & {fmt(m["top1"])} & {fmt(m["balanced_accuracy"])} & {fmt(m["top5"])} \\\\')
    lines+=['','DIRECT BASELINES AND VOCABULARY','Model  Frequency% Position% Train types Test coverage%']
    for key in ['qwen','llama']:
        r=combined[f'direct-{key}']
        lines.append(f'{key:<6} {fmt(r["baselines"]["frequency"]["top1"]):>10} {fmt(r["baselines"]["position"]["top1"]):>9} {r["training_token_types"]:>11} {fmt(r["results"]["down1"]["known_fraction"]):>14}')
    lines+=['','V-B: REAL GATHER FIDELITY (all rates in %)',
        'Platform Model Phase    Rows Gathers Raw FPR  Vote P  Vote R Filter P Filter R']
    detail=['ALL FIDELITY CONDITIONS','Platform Model Phase    Rows Regime             RawFPR  VoteP VoteR FilterP FilterR']
    for platform in ['intel','amd']:
        controls=read(ROOT/f'results/{platform}/controls/analysis.json');combined[f'{platform}-controls']=controls
        for key in ['qwen','llama']:
            r=read(ROOT/f'results/{platform}/live-{key}/analysis.json');combined[f'{platform}-live-{key}']=r
            for m in r['rows']:
                condition=m['condition'];budget={'single_execution':1,'full_replay':5,'stride_replay':25}[condition]
                vals=[fmt(m['raw']['fpr']),fmt(m['consensus']['precision']),fmt(m['consensus']['recall']),fmt(m['filtered']['precision']),fmt(m['filtered']['recall'])]
                detail.append(f'{platform:<8} {key:<5} {m["phase"]:<8} {m["observed"]:>4} {condition:<18} '+' '.join(f'{x:>7}' for x in vals))
                if condition=='full_replay' or (condition=='single_execution' and m['observed']!=1):continue
                lines.append(f'{platform:<8} {key:<5} {m["phase"]:<8} {m["observed"]:>4} {budget:>7} '+' '.join(f'{x:>7}' for x in vals))
                vb.append(f'{platform.capitalize()} & {key.capitalize()} & {m["phase"]} & {m["observed"]} & {budget} & {vals[0]} & {vals[3]} & {vals[4]} \\\\')
    detail+=['','LIVE FOOTPRINT DENSITY (single-execution, one observed row)', 'Platform Model Phase    Windows Empty Truth-positive groups%']
    for platform in ['intel','amd']:
        for key in ['qwen','llama']:
            for m in combined[f'{platform}-live-{key}']['rows']:
                if m['condition']=='single_execution' and m['observed']==1:
                    detail.append(f'{platform:<8} {key:<5} {m["phase"]:<8} {m["windows"]:>7} {m["empty_windows"]:>5} {fmt(m["raw"]["prevalence"]):>22}')
    detail+=['','SYNTHETIC FIXED-FOOTPRINT CONTROLS (four columns, all rates in %)', 'Platform Width Rows Obs Regime             RawFPR VoteP VoteR FilterP FilterR']
    for platform in ['intel','amd']:
        for m in combined[f'{platform}-controls']['rows']:
            if m['kind']!='fixed_columns':continue
            vals=[fmt(m['raw']['fpr']),fmt(m['consensus']['precision']),fmt(m['consensus']['recall']),fmt(m['filtered']['precision']),fmt(m['filtered']['recall'])]
            detail.append(f'{platform:<8} {m["width"]:>5} {m["rows"]:>4} {m["observed"]:>3} {m["condition"]:<18} '+' '.join(f'{v:>7}' for v in vals))
    detail+=['','TIMING CUTOFF VALIDATION (held-out primitive samples)', 'Platform Dataset  Cutoff ticks FPR% Recall%']
    for platform in ['intel','amd']:
        for dataset in ['controls','live-qwen','live-llama']:
            r=combined[f'{platform}-{dataset}'];m=r['calibration']
            detail.append(f'{platform:<8} {dataset:<10} {r["threshold"]:>10} {fmt(m["fpr"]):>6} {fmt(m["recall"]):>7}')
    detail+=['','DIRECT CLASSIFICATION DIAGNOSTICS', 'Model Group Known-top1% Outside-top20% Permuted-top1% Top1 95% CI']
    for key in ['qwen','llama']:
        for m in combined[f'direct-{key}']['results'].values():
            lo,hi=m['ci95_response_bootstrap']
            detail.append(f'{key:<6} {m["group_width"]:>4} {fmt(m["known_top1"]):>11} {fmt(m["outside_training_top20_top1"]):>14} {fmt(m["shuffled_test_features"]["top1"]):>14} [{100*lo:.2f}, {100*hi:.2f}]')
    lines+=['','NO-GATHER NEGATIVE CONTROLS (one observed row)', 'Platform Width Regime              N probes  FPR% First-probe FPR%']
    for platform in ['intel','amd']:
        for m in combined[f'{platform}-controls']['rows']:
            if m['kind']=='no_gather' and m['rows']==1:
                r=m['raw'];lines.append(f'{platform:<8} {m["width"]:>5} {m["condition"]:<18} {r["fp"]+r["tn"]:>9} {fmt(r["fpr"]):>7} {fmt(m["first_probe_per_flush"]["fpr"]):>16}')
    lines+=['','PAIRED DIRECT-TO-CACHE TOKEN CLASSIFICATION (same held-out decode cohort)',
        'Platform Model Regime              N  Direct% Consensus% Filtered% Filter gap(pp)']
    detail+=['','PAIRED CLASSIFICATION DETAILS', 'Platform Model Regime             Stage     Top1% Balanced% Top5% Gap pp [95% CI] Exact prefixes']
    for platform in ['intel','amd']:
        for key in ['qwen','llama']:
            r=read(ROOT/f'results/{platform}/bridge-{key}/results.json');combined[f'{platform}-bridge-{key}']=r
            for condition in ['single_execution','full_replay','stride_replay']:
                s={m['stage']:m for m in r['results'] if m['condition']==condition}
                lines.append(f'{platform:<8} {key:<5} {condition:<18} {s["direct"]["positions"]:>4} {fmt(s["direct"]["top1"]):>8} {fmt(s["consensus"]["top1"]):>10} {fmt(s["filtered"]["top1"]):>9} {s["filtered"]["observed_minus_direct_pp"]:>14.2f}')
                gathers={'single_execution':1,'full_replay':5,'stride_replay':25}[condition]
                bridge.append(f'{platform.capitalize()} & {key.capitalize()} & {gathers} & {s["direct"]["positions"]} & {fmt(s["direct"]["top1"])} & {fmt(s["consensus"]["top1"])} & {fmt(s["filtered"]["top1"])} \\\\')
                for stage,m in s.items():
                    lo,hi=m['paired_gap_ci95_pp']
                    detail.append(f'{platform:<8} {key:<5} {condition:<18} {stage:<9} {fmt(m["top1"]):>6} {fmt(m["balanced_accuracy"]):>9} {fmt(m["top5"]):>6} {m["observed_minus_direct_pp"]:>7.2f} [{lo:.2f}, {hi:.2f}] {m["exact_response_prefixes"]}/{m["responses"]}')
    lines+=['','Gathers=1 is one gather inside that measurement condition; other conditions are also tested on the same activation.',
        'Gathers=1/5 use full scans; Gathers=25 uses stride5 x5 repeats. This changes schedule and replay budget.',
        'Gathers=25 is replay fidelity, not single-execution recovery. Synchronized owned-process measurements; no NPU claim.',
        'Bridge uses the frozen direct-trained group32 classifier without cache retraining; at most16 responses x8 tokens.']
    (ROOT/'FINAL_RESULTS.txt').write_text('\n'.join(lines)+'\n');(ROOT/'DIAGNOSTICS.txt').write_text('\n'.join(detail)+'\n')
    (ROOT/'results.json').write_text(json.dumps(combined,indent=2)+'\n');paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    for name,cols,head,rows in [('va-decode','lrrrrr','Model & Group & Correct & Top-1 (\\%) & Balanced (\\%) & Top-5 (\\%)',va),
        ('vb-phase','lllrrrrr','CPU & Model & Phase & Rows & Gathers & Raw FPR (\\%) & Filtered P (\\%) & Filtered R (\\%)',vb),
        ('decode-bridge','llrrrrr','Trace CPU & Model & Gathers & Tokens & Direct (\\%) & Consensus (\\%) & Filtered (\\%)',bridge)]:
        (paper/f'{name}.tex').write_text('\\begin{tabular}{'+cols+'}\n\\hline\n'+head+' \\\\\n\\hline\n'+'\n'.join(rows)+'\n\\hline\n\\end{tabular}\n')
    print((ROOT/'FINAL_RESULTS.txt').read_text())

if __name__=='__main__':main()
