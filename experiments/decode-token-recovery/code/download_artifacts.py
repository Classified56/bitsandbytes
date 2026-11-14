"""Download, verify, and restore the published experiment records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checksums():
    return {name: sha for sha, name in
            (line.split('  ', 1) for line in (ROOT / 'ARTIFACT_SHA256SUMS').read_text().splitlines())}


def restore(path, expected, count):
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or member.name not in expected or member.name in seen):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            target = ROOT.joinpath(*relative.parts)
            if ROOT not in target.resolve().parents:
                raise ValueError(f'Path outside the experiment folder: {member.name}')
            seen.add(member.name)
            if target.exists():
                if digest(target) != expected[member.name]:
                    raise ValueError(f'Refusing to overwrite a changed file: {member.name}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                temporary = Path(f.name)
                try:
                    source = archive.extractfile(member)
                    with source:
                        shutil.copyfileobj(source, f)
                    f.flush()
                    if digest(temporary) != expected[member.name]:
                        raise ValueError(f'Checksum mismatch: {member.name}')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    if len(seen) != count:
        raise ValueError(f'Expected {count} files, found {len(seen)}')
    print(f'PASS: restored and verified {len(seen)} files from {path.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Use already downloaded release assets.')
    parser.add_argument('--only', choices=['observations', 'classifiers'])
    parser.add_argument('--verify-only', action='store_true', help='Verify restored records without downloading.')
    args = parser.parse_args()
    expected = checksums()
    if args.verify_only:
        selected = {p: h for p, h in expected.items()
                    if not args.only or (p.endswith('.pt') == (args.only == 'classifiers'))}
        for name, sha in selected.items():
            if digest(ROOT / name) != sha:
                raise ValueError(f'Checksum mismatch: {name}')
        print(f'PASS: {len(selected)} restored artifact checksums')
        return
    for asset in json.loads((ROOT / 'ARTIFACTS.json').read_text())['assets']:
        if args.only and args.only != asset['kind']:
            continue
        cache = args.from_dir or ROOT / '.artifacts'
        path = cache / asset['name']
        if not path.exists():
            if args.from_dir:
                raise FileNotFoundError(path)
            cache.mkdir(exist_ok=True)
            partial = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {asset["name"]} ({asset["bytes"]:,} bytes)', flush=True)
            with urllib.request.urlopen(asset['url'], timeout=120) as response, partial.open('wb') as f:
                shutil.copyfileobj(response, f)
            if partial.stat().st_size != asset['bytes'] or digest(partial) != asset['sha256']:
                raise ValueError(f'Download checksum mismatch: {partial}')
            partial.replace(path)
        if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}')
        restore(path, expected, asset['files'])


if __name__ == '__main__':
    main()
