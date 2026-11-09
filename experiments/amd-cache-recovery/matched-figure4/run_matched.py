"""Run the unmodified upstream Figure 4 protocol, preserving every acquisition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PINNED = "58cc1cc78f708c3a70c2f6cd969baa5f835ef957"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquisitions", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Verify source hashes and print commands without measuring or creating output")
    args = parser.parse_args()
    if args.acquisitions < 1:
        parser.error("acquisitions must be positive")
    base = Path(__file__).resolve().parent
    source, output = args.source.resolve(), args.output.resolve()
    config = json.loads((base / "matched-config.json").read_text())
    hashes = json.loads((base / "source-sha256.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    # Allow the archival commit (and later documentation-only commits). The
    # measurement/analysis files must still match the pinned protocol exactly.
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", *hashes], text=True)
    for filename, expected in hashes.items():
        actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    flags = []
    for key, value in config.items():
        flags.extend(["--" + key.replace("_", "-"), str(value)])
    commands = []
    for index in range(1, args.acquisitions + 1):
        destination = output / f"run-{index:02d}"
        commands.append([sys.executable, "-m", "prefill_flush_reload", *flags,
                         "--output", str(destination)])
    if args.dry_run:
        print(json.dumps({"source_commit": revision, "protocol_baseline_commit": PINNED,
                          "source_hashes_verified": len(hashes), "commands": commands}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": revision, "protocol_baseline_commit": PINNED,
        "source_hashes": hashes, "configuration": config,
        "primary_acquisition": "run-01", "planned_acquisitions": args.acquisitions,
        "policy": "First acquisition is primary; retain all later runs and failures; no tuning or replacement.",
        "commands": commands, "python": sys.version, "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "load_before": os.getloadavg(),
        "compiler": subprocess.check_output([os.environ.get("CC", "cc"), "--version"], text=True),
        "selected_environment": {key: os.environ.get(key) for key in
            ("CC", "CFLAGS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")},
        "native_build_flags": ["-O3", "-mf16c", "-shared", "-fPIC", "-lm"],
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "os_release": Path("/etc/os-release").read_text(), "runs": [],
    }
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    for index, command in enumerate(commands, 1):
        record = {"run": f"run-{index:02d}", "started_monotonic_ns": time.monotonic_ns(),
                  "started_unix_ns": time.time_ns(), "load_before": os.getloadavg()}
        print(f"START {record['run']}", flush=True)
        try:
            with (output / f"run-{index:02d}.log").open("w") as stream:
                completed = subprocess.run(command, cwd=source, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=120)
            record["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = 124
        record["elapsed_s"] = (time.monotonic_ns() - record["started_monotonic_ns"]) / 1e9
        summary_path = output / record["run"] / "summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            observed_config = {key:value for key,value in data["configuration"].items() if key != "output"}
            assert observed_config == config, observed_config
            measured = data["cases"]["injected_gather"]
            record.update({"cache_check_passed": data["cache_check_passed"],
                "probe_precision": measured["precision"], "probe_recall": measured["recall"],
                "probe_fpr": measured["false_positive_rate"],
                "groups": data["detection_summary"]["detected_groups"],
                "filtered_groups": data["detection_summary"]["group_detection"]["methods"]["consensus"]["leading_group"]["predicted_groups"]})
        manifest["runs"].append(record)
        save()
        print(json.dumps(record), flush=True)
        if record["returncode"] not in (0, 1) or not summary_path.exists():
            raise RuntimeError("Setup/numerical failure; retained logs; refusing to replace the run")
    manifest["load_after"] = os.getloadavg()
    manifest["complete"] = True
    save()


if __name__ == "__main__":
    main()
