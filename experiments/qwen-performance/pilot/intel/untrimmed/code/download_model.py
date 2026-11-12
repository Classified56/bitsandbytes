"""Download and verify the pinned official model; never execute model code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'provenance/model-source.json').read_text())
checkpoint = ROOT/'checkpoint'
checkpoint.mkdir(exist_ok=True)
records = []
for name, spec in manifest['files'].items():
    path = checkpoint/name
    url = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{name}"
    if not path.exists():
        print('Downloading '+name, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, path.with_suffix(path.suffix+'.part').open('wb') as output:
            while chunk := response.read(8*1024*1024):
                output.write(chunk)
        path.with_suffix(path.suffix+'.part').rename(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            digest.update(chunk)
    expected = spec.get('sha256')
    assert path.stat().st_size == spec['bytes'], name
    if expected:
        assert digest.hexdigest() == expected, name
    records.append({'file': name, 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
(ROOT/'provenance/checkpoint-verified.json').write_text(json.dumps(records, indent=2)+'\n')
print('Verified pinned checkpoint.', flush=True)
