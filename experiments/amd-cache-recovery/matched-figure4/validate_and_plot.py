"""Verify the preserved Figure 4 records, then plot without rerunning probes.

Requires matplotlib; the unmodified upstream analyzer is imported from --source.
All rates in JSON/CSV are fractions, and all group bounds are zero-based.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
PINNED = "58cc1cc78f708c3a70c2f6cd969baa5f835ef957"
CASES = ("no_outliers", "injected_gather", "injected_no_gather")
TRUTH = [192, 896, 1984, 2240]
SCHEDULE = ("case", "trial", "row", "column_start", "column_end_exclusive",
            "expected_access", "outlier_line", "sweep", "probe_index", "previous_probe_line")


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def score(tp, fp, fn, tn):
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None)


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                sample_stdev=statistics.stdev(values) if len(values) > 1 else None)


def validate(name, directory, config, analyze_csv, reference=None):
    summary = read_json(directory / "summary.json")
    assert {k: v for k, v in summary["configuration"].items() if k != "output"} == config, name
    with (directory / "lines.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9120, (name, len(rows))
    expected_keys = {(case, t, r, c) for case in CASES for t in range(5)
                     for r in range(4) for c in range(0, 4864, 32)}
    keys = [(r["case"], int(r["trial"]), int(r["row"]), int(r["column_start"])) for r in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys, name
    schedule = [[r[k] for k in SCHEDULE] for r in rows]
    schedule_hash = digest(json.dumps(schedule, separators=(",", ":")).encode())
    if reference is not None:
        assert schedule == reference["schedule"], name + ": probe schedule/labels differ"
        for case in CASES:
            for key in ("outlier_columns", "outlier_count", "observed_rows", "numerics"):
                assert summary["cases"][case].get(key) == reference["summary"]["cases"][case].get(key), (name, case, key)
    cutoff = summary["calibration"]["reload_threshold_tsc_ticks"]
    counts = {case: Counter() for case in CASES}
    for r in rows:
        case, start = r["case"], int(r["column_start"])
        pred, expected = int(r["predicted_hit"]), int(r["expected_access"])
        assert pred == int(int(r["latency_tsc_ticks"]) < cutoff), (name, r)
        assert expected == int(case == "injected_gather" and start in TRUTH), (name, r)
        assert int(r["outlier_line"]) == int(case != "no_outliers" and start in TRUTH), (name, r)
        counts[case][("tp" if expected else "fp") if pred else ("fn" if expected else "tn")] += 1
    raw = {}
    for case in CASES:
        raw[case] = score(*(counts[case][k] for k in ("tp", "fp", "fn", "tn")))
        for k, v in raw[case].items():
            assert v == summary["cases"][case][k], (name, case, k)
    analysis = analyze_csv(directory / "lines.csv", row_start=config["row_start"],
        probe_rows=config["probe_rows"], row_hit_fraction=config["row_hit_fraction"],
        min_row_fraction=config["min_row_fraction"], min_control_delta=config["min_control_delta"])
    saved = read_json(directory / "analysis.json")
    for field in ("observed_rows", "rules", "detected_groups", "control_adjusted_groups", "groups"):
        assert analysis[field] == saved[field], (name, field)
    if "group_detection" in saved:
        assert analysis["group_detection"] == saved["group_detection"], name
    # Independently recompute consensus/filter from raw CSV hits.
    hits = Counter((int(r["row"]), int(r["column_start"])) for r in rows
                   if r["case"] == "injected_gather" and int(r["predicted_hit"]))
    detected = [c for c in range(0, 4864, 32) if sum(hits[r, c] >= 3 for r in range(4)) >= 2]
    filtered = [c for c in detected if c - 32 not in detected]
    method = analysis["group_detection"]["methods"]["consensus"]
    for values, stage in ((detected, "raw"), (filtered, "leading_group")):
        result = method[stage]
        assert values == result["predicted_groups"], (name, stage)
        truth, pred = set(TRUTH), set(values)
        independent = score(len(truth & pred), len(pred - truth), len(truth - pred),
                            152 - len(truth | pred))
        for key in ("tp", "fp", "fn", "tn", "precision", "recall"):
            assert independent[key] == result[key], (name, stage, key)
    output = dict(name=name, raw_probe=raw, calibration=summary["calibration"],
        consensus=method["raw"], filtered=method["leading_group"],
        control_adjusted=analysis["group_detection"]["methods"]["control_adjusted"],
        spatial_offsets=analysis["spatial_offsets"],
        validation=dict(records=len(rows), configuration_match=True,
            probe_schedule_and_labels_match_intel=True, schedule_sha256=schedule_hash,
            classification_recomputed=True, consensus_and_filter_independently_recomputed=True,
            numerical_checks_match_intel=True,
            input_sha256={p.name: digest(p.read_bytes()) for p in directory.iterdir()
                          if p.is_file() and p.suffix in (".json", ".csv")}),
        cache_check_passed=summary["cache_check_passed"])
    return output, dict(summary=summary, schedule=schedule), analysis


def plot_comparison(data, plt):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True)
    fig.subplots_adjust(left=.19, right=.97, top=.84, bottom=.15, hspace=.55)
    fig.text(.04, .96, "Figure 4 replication: Intel and AMD", fontsize=24, weight="bold", va="top")
    fig.text(.04, .91, "64 × 4864 FP16  |  observed rows 0–3  |  seed 42  |  5 repeats  |  sweep / observed  |  stride 5",
             fontsize=11, color="#536177")
    for ax, record, title in zip(axes, [data["intel"], data["amd"][0]],
            ["Intel Core i7-10700 — original Figure 4 record", "AMD Ryzen 7 8700G — first acquisition (run-01)"]):
        sets = [TRUTH, record["consensus"]["predicted_groups"], record["filtered"]["predicted_groups"]]
        colors = ["#df7200", "#007db5", "#009d79"]
        for y, groups, color in zip([2, 1, 0], sets, colors):
            ax.broken_barh([(0, 4864)], (y - .24, .48), facecolors="#eff1f4")
            ax.broken_barh([(g, 32) for g in groups], (y - .24, .48), facecolors=color)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=25)
        ax.text(0, 1.04, f"Consensus: TP {record['consensus']['tp']}, FP {record['consensus']['fp']}  ·  "
                f"Filtered: TP {record['filtered']['tp']}, FP {record['filtered']['fp']}",
                fontsize=11, color="#536177", transform=ax.transAxes)
        ax.set_yticks([2, 1, 0], ["Ground truth", "Raw groups\n(after consensus)", "Filtered groups"], fontsize=12)
        ax.set_ylim(-.45, 2.45)
        ax.set_xlim(0, 4864)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 4864])
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", labelsize=11, length=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Activation column (32 FP16 columns per cache-line group)", fontsize=12, labelpad=10)
    fig.text(.04, .065, "Filter: retain the first group in each strictly consecutive candidate run. Each colored mark spans 32 columns.",
             fontsize=10, color="#536177")
    fig.text(.04, .035, "AMD: 10/10 acquisitions recover the exact group set after filtering; 5/10 have one false group before filtering.",
             fontsize=10, color="#536177")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(BASE / f"figure4-intel-amd.{suffix}", dpi=200, facecolor="white")
        if suffix == "svg":
            path = BASE / f"figure4-intel-amd.{suffix}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_repeats(data, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=.07, right=.98, top=.77, bottom=.22, wspace=.28)
    fig.text(.035, .96, "Same settings, ten AMD acquisitions", fontsize=21, weight="bold", va="top")
    fig.text(.035, .875, "Each acquisition includes five probe repeats. Run 01 was designated primary before execution.",
             fontsize=11, color="#536177")
    x = list(range(1, 11))
    raw = [100 * r["raw_probe"]["injected_gather"]["false_positive_rate"] for r in data["amd"]]
    intel = 100 * data["intel"]["raw_probe"]["injected_gather"]["false_positive_rate"]
    axes[0].plot(x, raw, "o-", color="#007db5", label="AMD measured")
    axes[0].axhline(intel, color="#df7200", ls="--", label="Intel original record")
    axes[0].set_ylabel("Probe false-positive rate (%)")
    axes[0].set_ylim(0, 11)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1].plot(x, [r["consensus"]["fp"] for r in data["amd"]], "o-", color="#007db5", label="AMD after consensus")
    axes[1].plot(x, [r["filtered"]["fp"] for r in data["amd"]], "s-", color="#009d79", label="AMD after filtering")
    axes[1].axhline(data["intel"]["consensus"]["fp"], color="#df7200", ls="--", label="Intel after consensus")
    axes[1].set_ylabel("False-positive groups")
    axes[1].set_ylim(-.3, 5.6)
    axes[1].set_yticks(range(6))
    axes[1].legend(frameon=False, fontsize=9, loc="center right")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel("AMD acquisition")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.035, .07, "Recall is 100% at every stage in both datasets. Intel also has zero false groups after filtering.",
             fontsize=10, color="#536177")
    fig.text(.035, .025, "These are repeated runs of one fixed input/seed, not ten different inputs or a matched ten-run Intel sample.",
             fontsize=10, color="#536177")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(BASE / f"figure4-amd-repeatability.{suffix}", dpi=200, facecolor="white")
        if suffix == "svg":
            path = BASE / f"figure4-amd-repeatability.{suffix}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def write_report(data):
    lines = [
        "Figure 4 replication on AMD with matched settings and comparison to the original Intel record",
        "Date: 2026-11-09",
        "",
        "Conclusion: all 10 acquisitions on the AMD Ryzen 7 8700G recovered the four true cache-line groups after the same filtering rule.",
        "Five were already exact after row consensus; five retained one adjacent false group, which the same filter removed.",
        "The first AMD acquisition had a higher raw probe FPR than the Intel record but fewer false groups after consensus.",
        "This establishes reproducibility for this synthetic input on AMD, not general AMD superiority or a uniquely identified prefetch cause.",
        "",
        "1. Baseline and code",
        f"Upstream: https://github.com/Classified56/bitsandbytes/tree/{PINNED}",
        "Branch: shadow_outlier; the link identifies the pinned commit. AMD measurement code was unchanged; 17 source files were checked by SHA256.",
        "Intel baseline: prefill_flush_reload/prefill-four-rows (word four, seed 42).",
        "This is distinct from prefill-4-rows (digit 4, seed 420) and the results labeled as Table III in the manuscript.",
        "Figure 4 uses four rows, seed 42, sweep / observed; its group positions match the selected record.",
        "Original Intel files were checked byte-for-byte against the upstream files. Originals are in intel-original/; rescored outputs are in intel-reanalysis/.",
        "Intel platform information comes from the manuscript: i7-10700, Ubuntu 22.04.5, 16 GB DDR4. No new Intel measurements were collected.",
        "Measured AMD environment: Ryzen 7 8700G, Ubuntu 24.04.4, Linux 7.0.0-38-generic, Python 3.12.3, GCC 13.3.0.",
        "AMD environment, affinity, compiler, load, commands, start times, and exit codes are recorded in amd/manifest.json.",
        "",
        "2. Matched parameters and measurement procedure",
        json.dumps(data["configuration"], indent=2, ensure_ascii=False),
        "",
        "64 x 4864 FP16 activations; four outlier columns at 204, 912, 2006, 2253; hidden dimension 896.",
        "The corresponding 32-column group starts are 192, 896, 1984, 2240; there are 152 candidate groups per row.",
        "The full 64-row gather executes each time; only rows 0-3 are flushed and reloaded.",
        "Each case has 5 trials; each trial is divided into 5 interleaved sweeps with stride 5 and a fixed-seed shuffle within each sweep.",
        "Case order: no_outliers, injected_gather, injected_no_gather.",
        "There are 3040 records per case and 9120 per acquisition; this is not a simultaneous snapshot of a single gather.",
        "Flush, gather, and reload are staged in one process in a controlled synthetic CPU experiment.",
        "The timing window precedes matrix multiplication and numerical verification, as in the original code. No NPU or real-LLM inference is executed here.",
        "Native build command: cc -O3 -mf16c -shared -fPIC kernels.c flush_reload_native.c -lm.",
        "",
        "The outlier activation threshold=32 is fixed. The cache-hit timing cutoff is a separate parameter, calibrated by the original code on a separate page:",
        "100 warmup samples, 2000 training samples to choose the cutoff, and 2000 independent holdout samples for verification. Case/outlier labels are not used for tuning.",
        "A hit satisfies latency_tsc_ticks < reload_threshold_tsc_ticks.",
        "Intel cutoff: 119 TSC ticks; primary AMD cutoff: 253; range across ten AMD acquisitions: 106-253.",
        "These different values are outputs of the same calibration algorithm, not changed experiment settings. TSC ticks are not core cycles or nanoseconds.",
        "No additional core pinning, CPU governor tuning, prefetcher disabling, or system tuning was applied.",
        "",
        "3. Three scoring stages (Raw groups in the figure is not raw probe classification)",
        "1. Raw probe: classify each reload before consensus; precision = TP/(TP+FP), FPR = FP/(FP+TN).",
        "2. Consensus: a row must hit in at least 3 of 5 repeats; at least 2 of 4 rows must qualify for the group to become a candidate.",
        "3. Filtered: keep the lowest-start group from each strictly consecutive candidate run with a 32-column step.",
        "The primary comparison uses the consensus selector, matching Figure 4, rather than the control_adjusted selector.",
        "Control-adjusted results are also fully retained in the JSON and original analysis files.",
        "Labels are used only for scoring, not filtering decisions. This heuristic can suppress real adjacent outlier groups; this input has none.",
        "",
        "4. Primary comparison: historical Intel record versus the predesignated first AMD acquisition",
        "Platform Stage          TP    FP    FN    TN    Precision    Recall    FPR",
    ]
    for name, record in (("Intel", data["intel"]), ("AMD", data["amd"][0])):
        for label, s in (("Raw probe", record["raw_probe"]["injected_gather"]),
                         ("Consensus", record["consensus"]), ("Filtered", record["filtered"])):
            fpr = s["fp"] / (s["fp"] + s["tn"])
            lines.append(f"{name:<6} {label:<12} {s['tp']:>4} {s['fp']:>5} {s['fn']:>5} {s['tn']:>5} "
                         f"{s['precision']:>11.4%} {s['recall']:>9.4%} {fpr:>9.4%}")
    lines += ["", "Group starts:"]
    for name, record in (("Intel", data["intel"]), ("AMD run-01", data["amd"][0])):
        lines += [f"{name} consensus: {record['consensus']['predicted_groups']}",
                  f"{name} filtered:  {record['filtered']['predicted_groups']}"]
    lines += ["", "Raw probe FPR in the control cases:", "Platform no_outliers   injected_no_gather"]
    for name, record in (("Intel", data["intel"]), ("AMD", data["amd"][0])):
        lines.append(f"{name:<6} {record['raw_probe']['no_outliers']['false_positive_rate']:.4%}       "
                     f"{record['raw_probe']['injected_no_gather']['false_positive_rate']:.4%}")
    lines += ["", "5. Ten AMD acquisitions (all retained, without selection or replacement)",
              "Run      Cutoff(ticks) Probe precision  Probe FPR   Consensus FP  Filtered FP"]
    for r in data["amd"]:
        s = r["raw_probe"]["injected_gather"]
        lines.append(f"{r['name']:<8} {r['calibration']['reload_threshold_tsc_ticks']:>10} "
                     f"{s['precision']:>16.4%} {s['false_positive_rate']:>10.4%} "
                     f"{r['consensus']['fp']:>14} {r['filtered']['fp']:>12}")
    a = data["amd_aggregate"]
    lines += ["", "Recall is 100% at every stage of every acquisition; filtered precision is always 100%.",
              f"Probe FPR: mean {a['false_positive_rate']['mean']:.4%}; "
              f"median {a['false_positive_rate']['median']:.4%}; "
              f"range {a['false_positive_rate']['minimum']:.4%}-{a['false_positive_rate']['maximum']:.4%}.",
              "These are short-session repeated measurements of the same seed/input, not ten inputs; there is no matched ten-acquisition Intel sample.",
              "Compared with the Intel record, most AMD acquisitions have more individual false hits but fewer false groups surviving row consensus.",
              "A single metric level therefore cannot establish overall platform superiority; no causal experiment identifies a particular prefetcher.",
              "", "6. Completed checks and limitations",
              "Verified 100320 records: 9120 Intel records and 91200 AMD records.",
              "Each acquisition matches the Intel baseline in configuration, case/trial/row/group coverage, complete probe order, and expected labels.",
              "Hits were recomputed from latencies, confusion matrices from CSV, and consensus/filter decisions independently; all match the archived results.",
              "AMD outlier columns and numerical verification metrics match the Intel record; all cache and numerical checks pass.",
              "Evidence is recorded in the validation fields of comparison.json and in amd/manifest.json.",
              "Historical Intel compiler versions, native binaries, and system-tuning details are unavailable; identical binaries and all system conditions cannot be guaranteed.",
              "The supported description is a cross-platform replication using the pinned upstream implementation and matching all recorded experiment parameters and probe order.",
              "This is not an experiment in which only the CPU changed while all other system factors were fully controlled.",
              "Overleaf was not modified.", "", "7. Files and reproduction",
              "figure4-intel-amd.{png,pdf,svg}: three group bands with the same meaning as Figure 4, comparing Intel and AMD.",
              "figure4-amd-repeatability.{png,pdf,svg}: variation across ten acquisitions and false positives before/after consensus and filtering.",
              "comparison.csv: 55 metric rows, with rates expressed as 0-1 fractions. comparison.json: full scoring and verification.",
              "amd/: original CSV, JSON, logs, and environment; intel-original/: original Intel baseline;",
              "intel-reanalysis/: Intel CSV rescored by the same upstream analysis function.",
              "This archive excludes manuscript source files; the original manuscript was not modified.",
              "", "Collect new measurements on Linux x86-64 (requires F16C, RDTSCP, 64-byte cache lines; output directory must not exist):",
              "  git clone https://github.com/Classified56/bitsandbytes.git upstream",
              f"  git -C upstream checkout --detach {PINNED}",
              "  python3 run_matched.py --source upstream --output amd-new --acquisitions 10",
              "", "Recompute archived results and regenerate figures without collecting new cache measurements:",
              "  python3 -m pip install matplotlib",
              "  python3 -B validate_and_plot.py --source upstream",
              "The analysis script reads amd/ and intel-original/ alongside the script. Organize new acquisitions in the same structure in a separate directory.",
              "Rebuilding rewrites derived plots and comparison files, but leaves the original amd/, intel-original/, and upstream code unchanged."]
    (BASE / "REPORT.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    hashes = read_json(BASE / "source-sha256.json")
    for name, sha in hashes.items():
        assert digest((source / name).read_bytes()) == sha, name
    for path in (BASE / "intel-original").iterdir():
        if path.is_file():
            assert path.read_bytes() == (source / "prefill_flush_reload" / "prefill-four-rows" / path.name).read_bytes(), path.name
    sys.path.insert(0, str(source))
    from prefill_flush_reload.analyze import analyze_csv, write_analysis
    config = read_json(BASE / "matched-config.json")
    intel, reference, analysis = validate("intel-original", BASE / "intel-original", config, analyze_csv)
    assert intel["consensus"]["predicted_groups"] == [192, 224, 896, 928, 960, 1984, 2240, 2272, 2304]
    write_analysis(analysis, BASE / "intel-reanalysis")
    manifest = read_json(BASE / "amd" / "manifest.json")
    assert manifest["complete"] and manifest["source_commit"] == PINNED
    assert manifest["primary_acquisition"] == "run-01" and manifest["planned_acquisitions"] == 10
    assert len(manifest["runs"]) == 10 and manifest["configuration"] == config
    assert manifest["source_hashes"] == hashes
    amd = []
    for index, run in enumerate(manifest["runs"], 1):
        assert run["run"] == f"run-{index:02d}"
        assert run["returncode"] == 0
        record, _, _ = validate(run["run"], BASE / "amd" / run["run"], config, analyze_csv, reference)
        assert record["cache_check_passed"]
        amd.append(record)
    aggregates = {key: stats([r["raw_probe"]["injected_gather"][key] for r in amd])
                  for key in ("precision", "recall", "false_positive_rate")}
    aggregates.update(consensus_exact_count=sum(r["consensus"]["exact_match"] for r in amd),
                      filtered_exact_count=sum(r["filtered"]["exact_match"] for r in amd),
                      consensus_precision=stats([r["consensus"]["precision"] for r in amd]),
                      calibrated_threshold_tsc_ticks=stats([r["calibration"]["reload_threshold_tsc_ticks"] for r in amd]))
    data = dict(source_commit=PINNED, configuration=config, primary_amd="run-01",
        intel_baseline="prefill_flush_reload/prefill-four-rows (seed 42; original recorded data)",
        intel=intel, amd=amd, amd_aggregate=aggregates,
        limitations=["Intel historical compiler/binary/OS tuning metadata are unavailable; cannot claim identical binaries or isolate CPU architecture causally.",
            "Intel has one historical acquisition; AMD repeats one fixed seed/input ten times in a short session.",
            "The code stages flush, gather and reload in one process; this is a controlled CPU synthetic footprint experiment, not an NPU or live-model attack.",
            "Reload cutoff is calibrated independently per acquisition by unchanged code; it is not forced to the same numeric value across platforms.",
            "The first-group heuristic can erase truly adjacent outlier groups; these four true groups are separated."])
    save_json(BASE / "comparison.json", data)
    with (BASE / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["platform", "acquisition", "stage", "case", "tp", "fp", "fn", "tn", "precision", "recall", "false_positive_rate", "predicted_groups"])
        for record in [intel, *amd]:
            platform = "Intel" if record is intel else "AMD"
            for case in CASES:
                values = record["raw_probe"][case]
                writer.writerow([platform, record["name"], "raw_probe", case,
                    *[values[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "false_positive_rate")], ""])
            for stage in ("consensus", "filtered"):
                values = record[stage]
                writer.writerow([platform, record["name"], stage, "injected_gather",
                    *[values[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")],
                    values["fp"] / (values["fp"] + values["tn"]), json.dumps(values["predicted_groups"])])
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(BASE / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_comparison(data, plt)
    plot_repeats(data, plt)
    write_report(data)
    print(json.dumps({"validated_acquisitions": 11, "validated_records": 11 * 9120,
                      "amd_aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
