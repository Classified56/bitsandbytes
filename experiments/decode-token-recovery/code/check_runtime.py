"""Verify the portable CPU backend by reproducing one archived Qwen forward."""
import argparse
from common import *
import bitsandbytes

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    backend=Path(bitsandbytes.__file__).resolve().parent
    assert backend==ROOT/'runtime/vendor/bitsandbytes',backend
    expected=json.loads((ROOT/'environment-intel.json').read_text())['backend_sha256']
    for name,h in expected.items():assert digest(backend/name)==h,name
    setup(1);tok,model=load('qwen',a.checkpoint,ROOT/'profiles/qwen.json')
    m=json.loads((ROOT/'results/intel/live-qwen/manifest.json').read_text());reference=m['validation'][0]
    out=model(input_ids=torch.tensor([reference['input_ids']]),use_cache=True,logits_to_keep=1)
    assert sha(out.logits)==reference['baseline_sha256'][0]
    assert int(out.logits[:,-1].argmax(-1))==reference['generated_ids'][0]
    result=dict(checks='PASS',backend_files=len(expected),bit_identical_prefill_logits=True,model='qwen',backend=str(backend))
    save(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
