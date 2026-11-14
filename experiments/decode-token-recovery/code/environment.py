"""Public runtime metadata only; no authentication or environment secrets."""
import argparse,hashlib,importlib.metadata as metadata,json,platform,subprocess,sys
from pathlib import Path
import torch,bitsandbytes
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
vendor=Path(bitsandbytes.__file__).resolve().parent
report=dict(python=sys.version,platform=platform.platform(),cpu=subprocess.check_output(['lscpu'],text=True),
    versions={n:metadata.version(n) for n in ['torch','transformers','numpy','safetensors']},bitsandbytes_version=bitsandbytes.__version__,
    bitsandbytes_path=str(vendor),torch_build=torch.__config__.show(),
    backend_sha256={str(f.relative_to(vendor)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(vendor.rglob('*')) if f.is_file() and f.suffix in {'.py','.so'}})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print('Saved',a.output)
