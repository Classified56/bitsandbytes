"""Generate or verify checksums for the portable experiment artifacts."""
import argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'checkpoint','checkpoints','__pycache__','.git','.matplotlib'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def included(path):
    rel=path.relative_to(ROOT)
    return path.is_file() and not set(rel.parts)&EXCLUDE and path.name not in {'SHA256SUMS','WORK_STATUS.txt'} and not path.name.startswith('.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();manifest=ROOT/'SHA256SUMS'
    if a.verify:
        count=0
        for line in manifest.read_text().splitlines():
            expected,relative=line.split('  ',1);path=(ROOT/relative).resolve()
            assert ROOT in path.parents and digest(path)==expected,relative
            count+=1
        print(f'PASS: {count} archived files match the manifest.')
    else:
        assert (ROOT/'verification.json').exists()
        files=sorted(p for p in ROOT.rglob('*') if included(p))
        manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in files))
        print(f'Archived {len(files)} files, {sum(p.stat().st_size for p in files)} bytes; model checkpoints excluded.')

if __name__=='__main__':main()
