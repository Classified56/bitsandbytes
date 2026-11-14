"""Export paper-review figures from independently verified experiment results."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/decode-extension-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    assert json.loads((ROOT/'verification.json').read_text())['checks']=='PASS'
    r=json.loads((ROOT/'results.json').read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,key in zip(axes,['qwen','llama']):
        data=r[f'direct-{key}'];rows=[data['results'][f'down{g}'] for g in [1,4,16,32]]
        y=np.array([100*m['top1'] for m in rows]);ci=100*np.array([m['ci95_response_bootstrap'] for m in rows])
        ax.errorbar(range(4),y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o-',capsize=4,color='#17698b',label='Direct features')
        for name,color,style in [('frequency','#777777',':'),('position','#bf6a2f','--')]:
            ax.axhline(100*data['baselines'][name]['top1'],label=f'{name.capitalize()} baseline',color=color,ls=style)
        ax.set(title='Qwen2.5-0.5B' if key=='qwen' else 'Llama-3.2-1B-Instruct',xticks=range(4),xticklabels=[1,4,16,32],xlabel='Columns per feature group',ylim=(0,100));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Held-out token Top-1 (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Direct decode classification: frozen training-only P99.9 profile')
    fig.text(.5,.01,'48 responses × up to 32 consumed tokens; response-bootstrap 95% intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'direct-classification.png');fig.savefig(out/'direct-classification.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for row,platform in enumerate(['intel','amd']):
        for col,key in enumerate(['qwen','llama']):
            ax=axes[row,col];data=r[f'{platform}-bridge-{key}']['results'];x=np.arange(3)
            for i,(stage,color) in enumerate([('direct','#17698b'),('consensus','#bd7c29'),('filtered','#8d4887')]):
                rows=[next(m for m in data if m['condition']==c and m['stage']==stage) for c in ['single_execution','full_replay','stride_replay']]
                y=np.array([100*m['top1'] for m in rows]);ci=100*np.array([m['ci95_response_bootstrap'] for m in rows]);pos=x+(i-1)*.24
                ax.bar(pos,y,width=.23,color=color,label=stage.capitalize(),yerr=[y-ci[:,0],ci[:,1]-y],capsize=2)
            ax.set(title=f'{platform.upper()} / {key.capitalize()}',xticks=x,xticklabels=['1 gather\nfull scan','5 gathers\nfull scan','25 gathers\nstride-5'],ylim=(0,100));ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel('Token Top-1 (%)')
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Frozen direct classifier applied to measured cache masks')
    fig.text(.5,.015,'Same held-out cohort within each panel; synchronized in-process measurements; 5/25 gathers require replay.\nResponse-bootstrap 95% intervals. One gather is one condition within the multi-condition instrumented run.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/'direct-to-cache.png');fig.savefig(out/'direct-to-cache.pdf');plt.close(fig)
    print('Saved scientific figures to',out)

if __name__=='__main__':main()
