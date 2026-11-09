"""Verify archived CPU measurements and write one manuscript results TXT.

No measurements are executed. The source hashes and all raw CSV classifications
are checked before the report is written. Percentages are labeled explicitly.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
PINNED = "58cc1cc78f708c3a70c2f6cd969baa5f835ef957"
CASES = ("no_outliers", "injected_gather", "injected_no_gather")


def read(path):
    return json.loads(path.read_text())


def percentage(value):
    return "NA" if value is None else f"{100 * value:.6f}"


def validate(path, analyze_csv):
    summary = read(path / "summary.json")
    cfg = summary["configuration"]
    threshold = summary["calibration"]["reload_threshold_tsc_ticks"]
    starts = list(range(0, cfg["intermediate"], 32))
    rows = range(cfg["row_start"], cfg["row_start"] + cfg["probe_rows"])
    actual = {c // 32 * 32 for c in summary["cases"]["injected_gather"]["outlier_columns"]}
    counts = {case: Counter() for case in CASES}
    seen = set()
    with (path / "lines.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            case, row, trial, start = r["case"], int(r["row"]), int(r["trial"]), int(r["column_start"])
            key = (case, trial, row, start)
            assert key not in seen, (path, key)
            seen.add(key)
            assert case in CASES and row in rows and start in starts and 0 <= trial < cfg["repeats"]
            assert int(r["column_end_exclusive"]) == start + 32
            expected, hit = int(r["expected_access"]), int(r["predicted_hit"])
            assert expected == int(case == "injected_gather" and start in actual)
            assert int(r["outlier_line"]) == int(case != "no_outliers" and start in actual)
            assert hit == int(int(r["latency_tsc_ticks"]) < threshold)
            counts[case][("tp" if expected else "fp") if hit else ("fn" if expected else "tn")] += 1
    assert len(seen) == 3 * cfg["repeats"] * cfg["probe_rows"] * len(starts)
    for case in CASES:
        c, s = counts[case], summary["cases"][case]
        for key in ("tp", "fp", "fn", "tn"):
            assert c[key] == s[key], (path, case, key)
        ratios = {"precision": (c["tp"], c["tp"] + c["fp"]),
                  "recall": (c["tp"], c["tp"] + c["fn"]),
                  "false_positive_rate": (c["fp"], c["fp"] + c["tn"])}
        for key, (num, den) in ratios.items():
            assert (num / den if den else None) == s[key], (path, case, key)
    analysis = analyze_csv(path / "lines.csv", **{k: cfg[k] for k in
        ("row_start", "probe_rows", "row_hit_fraction", "min_row_fraction", "min_control_delta")})
    saved = read(path / "analysis.json")
    for key in ("groups", "detected_groups", "control_adjusted_groups", "spatial_offsets", "probe_order_diagnostic"):
        assert saved[key] == analysis[key], (path, key)
    if "group_detection" in saved:
        assert saved["group_detection"] == analysis["group_detection"], path
    return dict(path=str(path.relative_to(BASE)), summary=summary, analysis=analysis, records=len(seen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    hashes = read(BASE / "matched-figure4/source-sha256.json")
    for name, expected in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(source))
    from prefill_flush_reload.analyze import analyze_csv
    records = {"Intel": validate(BASE / "matched-figure4/intel-original", analyze_csv)}
    for index in range(1, 11):
        records[f"AMD-{index:02d}"] = validate(BASE / "matched-figure4/amd" / f"run-{index:02d}", analyze_csv)
    supplementary = sorted((BASE / "supplementary-20261107/results").glob("prefill-*"))
    for path in supplementary:
        records[path.name] = validate(path, analyze_csv)
    assert len(records) == 15
    total = sum(r["records"] for r in records.values())
    assert total == 405840, total
    amd = [records[f"AMD-{i:02d}"] for i in range(1, 11)]
    lines = [
        "AMD CPU CACHE-RECOVERY: CONSOLIDATED FINAL RESULTS",
        "Archive dates: 2026-11-07 (supplementary), 2026-11-09 (matched Figure 4)",
        "",
        "This file consolidates the final results for manuscript use. The primary dataset",
        "contains ten AMD acquisitions matching Figure 4. Four earlier CPU cache configurations",
        "are reported separately as supplementary results; their success rates are not pooled.",
        "All tables are recomputed from the raw CSV/JSON records. This archive contains no",
        "real-LLM, NPU, or end-to-end classification results.",
        "",
        "1. Scope, provenance, and manuscript-ready conclusion",
        "",
        "On an AMD Ryzen 7 8700G, all ten acquisitions matching the supplied Figure 4",
        "configuration recovered the four injected 32-column groups after the original",
        "contiguous-run filter (precision = recall = 100%). Before filtering, row",
        "consensus yielded no false groups in five acquisitions and one adjacent false",
        "group in each of the other five. Raw probe FPR ranged from 5.101351% to",
        "9.358108% (mean 8.162162%). The historical Intel i7-10700 record had a raw",
        "probe FPR of 5.810811%, five false groups after consensus, and exact recovery",
        "after filtering. The protocol therefore transfers to this AMD platform for",
        "the tested synthetic input, while probe-level noise and group-level errors",
        "differ between the recorded environments. These data do not establish a causal",
        "CPU-architecture effect or generalization to real LLM workloads.",
        "",
        f"Measurement/analysis baseline commit: {PINNED}",
        "Repository: https://github.com/Classified56/bitsandbytes (shadow_outlier branch)",
        "Intel record: prefill_flush_reload/prefill-four-rows, seed 42 (not prefill-4-rows, seed 420).",
        "Intel hardware/OS from manuscript: i7-10700, Ubuntu 22.04.5 LTS, 16 GB DDR4.",
        "AMD: Ryzen 7 8700G, 8 cores/16 threads, 64-byte cache lines; Ubuntu 24.04.4;",
        "Linux 7.0.0-38-generic; Python 3.12.3; GCC 13.3.0. Full metadata: matched-figure4/amd/manifest.json.",
        "Native build flags: -O3 -mf16c -shared -fPIC; linked with -lm.",
        "Measurement files were unchanged and verified by 17 SHA256 hashes. No extra",
        "core pinning, governor tuning, prefetcher modification, or threshold tuning was applied.",
        "The wrapper records every planned acquisition, including failure, and designates",
        "run-01 as primary before starting. The archived ten runs all exited with code 0.",
        "",
        "2. Matched settings and metric definitions",
        "",
        json.dumps(read(BASE / "matched-figure4/matched-config.json"), indent=2),
        "True FP16 column indices: 204, 912, 2006, 2253 (zero-based).",
        "True 32-column group starts: 192, 896, 1984, 2240; group ends are exclusive.",
        "FP16 activation matrix: 64 x 4864; hidden dimension 896; 152 candidate groups per row.",
        "The complete 64-row gather executes, but only rows 0-3 are flushed and observed.",
        "Each case has 5 trials; each trial uses 5 interleaved, shuffled sweeps (stride 5).",
        "Case order: no_outliers, injected_gather, injected_no_gather.",
        "Each case has 3040 probes: gather case has 80 expected-positive and 2960 expected-negative probes.",
        "The two controls have 3040 expected-negative probes each. Total: 9120 probes per acquisition.",
        "This reconstructs a footprint across repeated fresh flush/gather windows, not one simultaneous snapshot.",
        "The flush/gather/reload sequence is staged in one process, before matrix multiplication and numerical checks.",
        "",
        "Raw probe: threshold each reload; precision=TP/(TP+FP), recall=TP/(TP+FN), FPR=FP/(FP+TN).",
        "Consensus: a row hits on >=3/5 repeats; a group is selected if >=2/4 rows hit consistently.",
        "Filtered: retain only the first group of each strictly contiguous candidate run (step=32 columns).",
        "Raw groups in Figure 4 means AFTER consensus, not individual probes. Group scoring has 4 positives/148 negatives.",
        "Control-adjusted (supplemental selector): also require hit fraction to exceed each control by >=0.5 per row.",
        "All percentages below are 0-100, not 0-1. NA means an undefined metric (e.g., recall in all-negative controls).",
        "Group FPR and probe FPR use different units/denominators and must not be interchanged.",
        "",
        "3. Primary matched comparison: historical Intel versus AMD acquisition 01",
        "",
        "platform stage          TP FP FN TN precision_pct recall_pct FPR_pct F1_pct",
    ]
    for name in ("Intel", "AMD-01"):
        r = records[name]
        method = r["analysis"]["group_detection"]["methods"]["consensus"]
        for stage, s in (("probe", r["summary"]["cases"]["injected_gather"]),
                         ("consensus", method["raw"]), ("filtered", method["leading_group"])):
            fpr = s["fp"] / (s["fp"] + s["tn"])
            f1 = 2*s["tp"] / (2*s["tp"] + s["fp"] + s["fn"])
            lines.append(f"{name:8s} {stage:12s} {s['tp']} {s['fp']} {s['fn']} {s['tn']} "
                         f"{percentage(s['precision'])} {percentage(s['recall'])} {percentage(fpr)} {percentage(f1)}")
        lines.append(f"{name} consensus group starts: {method['raw']['predicted_groups']}")
        lines.append(f"{name} filtered group starts: {method['leading_group']['predicted_groups']}")
    lines += ["", "4. All matched raw-probe results, including both controls", "",
              "acquisition case TP FP FN TN precision_pct recall_pct FPR_pct"]
    for name in ("Intel", *[f"AMD-{i:02d}" for i in range(1, 11)]):
        for case in CASES:
            s = records[name]["summary"]["cases"][case]
            lines.append(f"{name} {case} {s['tp']} {s['fp']} {s['fn']} {s['tn']} "
                         f"{percentage(s['precision'])} {percentage(s['recall'])} {percentage(s['false_positive_rate'])}")
    lines += ["", "AMD descriptive statistics across ten acquisitions of the SAME input:",
              "case metric mean_pct sample_sd_percentage_points median_pct min_pct max_pct"]
    for case in CASES:
        for metric in ("precision", "recall", "false_positive_rate"):
            values = [r["summary"]["cases"][case][metric] for r in amd]
            if any(v is None for v in values):
                continue
            lines.append(f"{case} {metric} " + " ".join(percentage(v) for v in
                (statistics.mean(values), statistics.stdev(values), statistics.median(values), min(values), max(values))))
    lines += ["Statistics are acquisition-level means/SD, not pooled precision. No independence or population confidence interval is assumed.",
              "", "5. All matched group recovery results", "",
              "acquisition selector stage TP FP FN TN precision_pct recall_pct F1_pct exact_set predicted_group_starts"]
    for name in ("Intel", *[f"AMD-{i:02d}" for i in range(1, 11)]):
        for selector, method in records[name]["analysis"]["group_detection"]["methods"].items():
            for stage in ("raw", "leading_group"):
                s = method[stage]
                lines.append(f"{name} {selector} {stage} {s['tp']} {s['fp']} {s['fn']} {s['tn']} "
                             f"{percentage(s['precision'])} {percentage(s['recall'])} {percentage(s['f1'])} "
                             f"{s['exact_match']} {s['predicted_groups']}")
    for selector in ("consensus", "control_adjusted"):
        for stage in ("raw", "leading_group"):
            scores = [r["analysis"]["group_detection"]["methods"][selector][stage] for r in amd]
            lines.append(f"AMD {selector}/{stage}: exact sets {sum(s['exact_match'] for s in scores)}/10; "
                         f"mean precision {percentage(statistics.mean(s['precision'] for s in scores))}%; "
                         f"mean recall {percentage(statistics.mean(s['recall'] for s in scores))}%.")
    lines += ["", "6. Calibration and observed latency distributions", "",
              "Cache cutoff is chosen on a separate page: 100 warmup samples, 2000 training and 2000 holdout samples.",
              "It never uses the case/outlier ground truth. Activation threshold=32 is a separate, fixed parameter.",
              "TSC ticks are not core cycles or nanoseconds. Cross-platform timing differences are not latency speedups.",
              "dataset cutoff_TSC_ticks heldout_TP FP FN TN balanced_accuracy_pct cache_check_passed"]
    for name, r in records.items():
        s = r["summary"]["calibration"]
        lines.append(f"{name} {s['reload_threshold_tsc_ticks']} {s['tp']} {s['fp']} {s['fn']} {s['tn']} "
                     f"{percentage(s['balanced_accuracy'])} {r['summary']['cache_check_passed']}")
    lines += ["", "Gather-case latency distribution; labels are ground truth, not predicted cache state.",
              "dataset expected_class n p05_TSC_ticks median_TSC_ticks p95_TSC_ticks"]
    for name, r in records.items():
        for label, s in r["summary"]["cases"]["injected_gather"]["latencies_tsc_ticks"].items():
            lines.append(f"{name} {label} {s['n']} {s['p05']} {s['median']} {s['p95']}")
    lines += ["", "7. Spatial diagnostic around the known true groups (supplementary observation)",
              "", "Offsets are measured in cache-line groups of 32 columns; positive is toward larger column indices.",
              "Truth is used only to describe this diagnostic, never to decide which detected groups to remove.",
              "Each offset/case pools probes at that offset from all true groups and observed rows within one acquisition.",
              "offset_groups case Intel_hits/observations Intel_hit_pct AMD_mean_hit_pct AMD_min_hit_pct AMD_max_hit_pct"]
    for index, item in enumerate(records["Intel"]["analysis"]["spatial_offsets"]):
        for case in CASES:
            s = item[case]
            values = [r["analysis"]["spatial_offsets"][index][case]["hit_fraction"] for r in amd]
            lines.append(f"{item['offset_groups']} {case} {s['hits']}/{s['observations']} "
                         + " ".join(percentage(v) for v in
                                    (s["hit_fraction"], statistics.mean(values), min(values), max(values))))
    lines += ["This spatial pattern does not identify a hardware prefetcher causally. No prefetcher-disable or cache-level isolation experiment was performed.",
              "", "8. Earlier AMD configurations: descriptive supplementary results, NOT matched Figure 4", "",
              "Each configuration below has one acquisition with five probe repeats. They are not an additional matched four-run cohort.",
              "They share 64 x 4864 FP16, hidden=896, 4 outliers, activation threshold=32 and the same decision fractions.",
              "Multiple settings differ simultaneously; do not claim a one-factor ablation or controlled speed/accuracy improvement."]
    for path in supplementary:
        name, r = path.name, records[path.name]
        lines += ["", name, "configuration=" + json.dumps({k: v for k, v in r["summary"]["configuration"].items() if k != "output"}, sort_keys=True),
                  "true_columns=" + json.dumps(r["summary"]["cases"]["injected_gather"]["outlier_columns"]),
                  "case TP FP FN TN precision_pct recall_pct FPR_pct"]
        for case in CASES:
            s = r["summary"]["cases"][case]
            lines.append(f"{case} {s['tp']} {s['fp']} {s['fn']} {s['tn']} "
                         f"{percentage(s['precision'])} {percentage(s['recall'])} {percentage(s['false_positive_rate'])}")
        for selector, method in r["analysis"]["group_detection"]["methods"].items():
            for stage in ("raw", "leading_group"):
                s = method[stage]
                lines.append(f"{selector}/{stage}: TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']} "
                             f"precision={percentage(s['precision'])}% recall={percentage(s['recall'])}% "
                             f"F1={percentage(s['f1'])}% exact_set={s['exact_match']} groups={s['predicted_groups']}")
    lines += ["", "All four supplementary acquisitions are exact after consensus and after filtering.",
              "The single-probe configuration re-flushes and re-gathers for every target line, so its zero observed errors",
              "cannot be compared as a free improvement over sweep probing. Timing overhead was not benchmarked.",
              "", "9. Numerical validation (supporting functional checks, not cache-recovery accuracy)", "",
              "Storage: original activations/weights FP16, quantized tensors INT8, scales FP32;",
              "main accumulation INT32, correction FP32, output rounding FP16.",
              "Each distinct numerical report is printed once, with all datasets sharing it."]
    numeric = defaultdict(list)
    for name, r in records.items():
        for case in CASES:
            if "numerics" in r["summary"]["cases"][case]:
                numeric[json.dumps(r["summary"]["cases"][case]["numerics"], sort_keys=True)].append(f"{name}/{case}")
    for serialized, names in numeric.items():
        lines += ["", "Datasets: " + ", ".join(names), json.dumps(json.loads(serialized), indent=2)]
    lines += ["Errors are normal/mixed arithmetic versus the stated reconstructed/original-weight references.",
              "Relative L2 is a ratio, not a percentage in the JSON above. These checks do not measure downstream model/task accuracy.",
              "", "10. Validation and limits on claims", "",
              f"Verified {total} raw records across {len(records)} datasets (1 Intel, 10 matched AMD, 4 supplemental AMD).",
              "Each CSV was checked for complete/unique case-trial-row-group coverage, correct truth labels, and",
              "prediction equality to latency < calibrated cutoff. Confusion matrices and rates were recomputed.",
              "Row/group analysis, spatial offsets and probe-order diagnostics were recomputed with the pinned analyzer",
              "and compared to archived reports. The matched validator additionally checks the complete probe order",
              "against Intel and independently recomputes consensus/filter decisions. Raw input files are unmodified.",
              "",
              "Limits:",
              "- Intel is one historical acquisition. Its original compiler/binary, tuning and scheduler state are unavailable.",
              "- AMD repeats the same seed/input ten times in one short session; these are not ten independent inputs.",
              "- No general claim of AMD superiority, robust real-world attack success, or statistically isolated CPU effect follows.",
              "- The first-group filter may erase genuinely adjacent outlier groups. This input has separated true groups.",
              "- The results identify 32-column cache-line groups, not individual FP16 feature columns.",
              "- The controlled single-process experiment does not establish cross-process exploitation or NPU leakage.",
              "- No real-model recovery, end-to-end classification accuracy, or first-token latency claim is measured here.",
              "- Zero observed errors are finite-sample findings, not a proof of a zero population error rate.",
              "- No causal prefetch mechanism, cache-level attribution, or architectural performance advantage is established.",
              "", "11. Artifact map and regeneration", ""]
    for name, r in records.items():
        lines.append(f"{name}: {r['path']} ({r['records']} records)")
    lines += ["", "Primary comparison: matched-figure4/figure4-intel-amd.{png,pdf,svg}",
              "Repeatability: matched-figure4/figure4-amd-repeatability.{png,pdf,svg}",
              "Machine-readable matched metrics: matched-figure4/comparison.{json,csv}",
              "Commands from repository root:",
              "  python3 -B experiments/amd-cache-recovery/matched-figure4/validate_and_plot.py --source .",
              "  python3 -B experiments/amd-cache-recovery/summarize_final_results.py --source .",
              "Only the first command needs Matplotlib. Neither command executes cache measurements.",
              "Raw per-run reports contain additional per-row and probe-order detail, available for future analysis.",
              "This TXT consolidates the final evaluated outcomes; per-probe raw observations remain in CSV."]
    (BASE / "FINAL_RESULTS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"validated_datasets": len(records), "validated_raw_records": total,
                      "results_file": str(BASE / "FINAL_RESULTS.txt")}, indent=2))


if __name__ == "__main__":
    main()
