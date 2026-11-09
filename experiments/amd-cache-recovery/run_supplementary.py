"""Re-run the four archived supplemental CPU configurations without tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    hashes = json.loads((BASE / "matched-figure4/source-sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    configs = {}
    for path in sorted((BASE / "supplementary-20261107/results").glob("*/summary.json")):
        configs[path.parent.name] = {k: v for k, v in json.loads(path.read_text())["configuration"].items() if k != "output"}
    assert len(configs) == 4
    commands = {}
    for name, config in configs.items():
        flags = [value for k, v in config.items() for value in ("--" + k.replace("_", "-"), str(v))]
        commands[name] = [sys.executable, "-m", "prefill_flush_reload", *flags, "--output", str(output / name)]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(configurations=configs, commands=commands, source_hashes=hashes,
                    source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                    python=sys.version, platform=platform.platform(),
                    policy="Preserve all four runs and failures, no tuning or replacement.", runs=[])
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    failed = False
    for name, command in commands.items():
        started = time.time_ns()
        with (output / (name + ".log")).open("w") as log:
            try:
                status = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
            except subprocess.TimeoutExpired:
                status = 124
        manifest["runs"].append(dict(name=name, started_unix_ns=started, returncode=status))
        failed |= status != 0
        save()
        print(name, status, flush=True)
    manifest["complete"] = True
    save()
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
