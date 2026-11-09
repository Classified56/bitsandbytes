"""Verify archived CPU measurements and write one manuscript results TXT.

No measurements are executed. The source hashes and all raw CSV classifications
are checked before the report is written. Percentages are labeled explicitly.
"""
import argparse
from collections import Counter
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
    return "NA" if value is None else f"{100 * value:.3f}"


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



def table(headers, rows):
    """Aligned ASCII tables remain readable in GitHub's plain-text viewer."""
    cells = [[str(value) for value in row] for row in [headers, *rows]]
    assert all(len(row) == len(headers) for row in cells)
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    def format_row(row):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
    return [format_row(cells[0]), "|" + "|".join("-" * (width + 2) for width in widths) + "|",
            *[format_row(row) for row in cells[1:]]]


def render_results(records):
    intel = records["Intel"]
    amd = [records[f"AMD-{i:02d}"] for i in range(1, 11)]
    primary = amd[0]
    supplementary = [("S1", "prefill-default"), ("S2", "prefill-four-rows"),
                     ("S3", "prefill-seed43"), ("S4", "prefill-two-rows-single")]
    group = lambda r, selector="consensus", stage="raw": r["analysis"]["group_detection"]["methods"][selector][stage]
    probe = lambda r, case="injected_gather": r["summary"]["cases"][case]
    span = lambda values: str(min(values)) if min(values) == max(values) else f"{min(values)}-{max(values)}"
    lines = ["AMD CPU CACHE-RECOVERY: FINAL RESULTS", "Matched: 2026-11-09 | Supplementary: 2026-11-07",
             "P=precision; R=recall; FPR=false-positive rate. Rates: %; relative L2: ratio; latency: TSC ticks.", ""]

    def section(title, headers, rows, note=None):
        lines.extend([title, *table(headers, rows)])
        if note:
            lines.append(note)
        lines.append("")

    section("1. Matched protocol", ["Item", "Setting"], [
        ("Platforms", "Intel i7-10700 (historical); AMD Ryzen 7 8700G"),
        ("Input", "64 x 4864 FP16; hidden=896; 4 outliers; activation threshold=32"),
        ("Acquisitions", "Intel: 1; AMD: 10; AMD-01 designated primary before execution"),
        ("Sampling", "Seed=42; rows=0-3; repeats=5; stride=5; sweep / observed"),
        ("Counts per acquisition", "Gather: 80 positive + 2960 negative probes; each control: 3040 negative"),
        ("Group scoring", "152 groups: 4 positive + 148 negative; 32 FP16 columns/group"),
        ("Consensus / filtering", ">=3/5 hits in >=2/4 rows; retain first group of each adjacent run"),
        ("Control-adjusted selector", "Additionally require row hit fraction >= each control + 0.5"),
        ("Timing cutoff", "Auto-calibrated; 100 warmup / 2000 training / 2000 holdout samples"),
        ("True group starts", "192, 896, 1984, 2240"),
    ])

    rows = []
    for name, record in (("Intel", intel), ("AMD-01", primary)):
        for stage, s in (("Probe", probe(record)), ("Consensus", group(record)),
                         ("Filtered", group(record, stage="leading_group"))):
            fpr = s["fp"] / (s["fp"] + s["tn"])
            f1 = 2*s["tp"] / (2*s["tp"] + s["fp"] + s["fn"])
            rows.append([name, stage, *[s[k] for k in ("tp", "fp", "fn", "tn")],
                         percentage(s["precision"]), percentage(s["recall"]), percentage(fpr), percentage(f1)])
    section("2. Primary Intel/AMD comparison", ["Platform", "Stage", "TP", "FP", "FN", "TN", "P%", "R%", "FPR%", "F1%"], rows,
            "Probe FPR and group FPR have different denominators. Figure 4 'Raw groups' means after consensus.")

    rows = []
    for name in ["Intel", *[f"AMD-{i:02d}" for i in range(1, 11)]]:
        r = records[name]
        rows.append([name, r["summary"]["calibration"]["reload_threshold_tsc_ticks"],
                     percentage(probe(r)["precision"]), percentage(probe(r)["false_positive_rate"]),
                     percentage(probe(r, "no_outliers")["false_positive_rate"]),
                     percentage(probe(r, "injected_no_gather")["false_positive_rate"]),
                     group(r)["fp"], group(r, "control_adjusted")["fp"], group(r, stage="leading_group")["fp"]])
    section("3. Every matched acquisition", ["Run", "Cutoff", "Probe P%", "Probe FPR%", "C0 FPR%", "C1 FPR%", "Cons. FP", "Adj. FP", "Filt. FP"], rows,
            "C0=no_outliers; C1=injected_no_gather. Recall=100% throughout; filtered P/R=100% for both selectors.")

    rows = []
    for label, values in [
        ("Probe precision", [probe(r)["precision"] for r in amd]),
        ("Probe FPR", [probe(r)["false_positive_rate"] for r in amd]),
        ("C0 FPR", [probe(r, "no_outliers")["false_positive_rate"] for r in amd]),
        ("C1 FPR", [probe(r, "injected_no_gather")["false_positive_rate"] for r in amd]),
        ("Consensus precision", [group(r)["precision"] for r in amd]),
    ]:
        rows.append([label, *[percentage(v) for v in (statistics.mean(values), statistics.stdev(values),
                    statistics.median(values), min(values), max(values))]])
    section("4. AMD repeatability (10 acquisitions, same input)", ["Metric", "Mean%", "SD (pp)", "Median%", "Min%", "Max%"], rows)
    rows = []
    for selector in ("consensus", "control_adjusted"):
        for stage, label in (("raw", "Before filter"), ("leading_group", "After filter")):
            scores = [group(r, selector, stage) for r in amd]
            rows.append([selector, label, f"{sum(s['exact_match'] for s in scores)}/10",
                         percentage(statistics.mean(s["precision"] for s in scores)),
                         percentage(statistics.mean(s["recall"] for s in scores))])
    lines.extend(table(["Selector", "Stage", "Exact sets", "Mean P%", "Mean R%"], rows) + [""])

    rows = []
    for label, cohort in [("Intel", [intel]), ("AMD (10)", amd),
                          *[(sid, [records[name]]) for sid, name in supplementary]]:
        values = []
        for expected in ("expected_access", "expected_no_access"):
            values.append(" / ".join(span([probe(r)["latencies_tsc_ticks"][expected][key] for r in cohort])
                                     for key in ("p05", "median", "p95")))
        rows.append([label, span([r["summary"]["calibration"]["reload_threshold_tsc_ticks"] for r in cohort]), *values])
    section("5. Timing / calibration", ["Dataset", "Cutoff", "Expected access: p05 / med / p95", "Expected no-access: p05 / med / p95"], rows,
            "Ranges span acquisitions. All 15 datasets: holdout balanced accuracy=100%; cache/scalar checks passed.")

    rows = []
    for sid, name in supplementary:
        cfg = records[name]["summary"]["configuration"]
        rows.append([sid, name, cfg["seed"], f"{cfg['row_start']}-{cfg['row_start']+cfg['probe_rows']-1}",
                     cfg["probe_mode"], cfg["flush_scope"], cfg["probe_stride"]])
    section("6. Supplementary AMD configurations (one acquisition each)", ["ID", "Directory", "Seed", "Rows", "Mode", "Flush", "Stride"], rows,
            "Shared: shape/hidden size/activation threshold/repeats; row/group fractions=0.6/0.5.\n"
            "S3 true group starts: 288, 1152, 2336, 3776; others use seed-42 groups.")
    rows = []
    for sid, name in supplementary:
        r, s = records[name], probe(records[name])
        rows.append([sid, s["tp"], s["fp"], s["fn"], percentage(s["precision"]), percentage(s["recall"]),
                     percentage(s["false_positive_rate"]), percentage(probe(r, "no_outliers")["false_positive_rate"]),
                     percentage(probe(r, "injected_no_gather")["false_positive_rate"]), group(r)["fp"],
                     group(r, stage="leading_group")["fp"]])
    lines.extend(table(["ID", "TP", "FP", "FN", "Probe P%", "Probe R%", "Probe FPR%", "C0 FPR%", "C1 FPR%", "Cons. FP", "Filt. FP"], rows))
    lines.extend(["Both selectors recover all four groups in S1-S4 before/after filtering (P/R=100%).", ""])

    rows = []
    for index, item in enumerate(intel["analysis"]["spatial_offsets"]):
        values = [r["analysis"]["spatial_offsets"][index]["injected_gather"]["hit_fraction"] for r in amd]
        means = [statistics.mean(r["analysis"]["spatial_offsets"][index][case]["hit_fraction"] for r in amd)
                 for case in CASES]
        rows.append([item["offset_groups"], item["injected_gather"]["observations"],
                     *[percentage(item[c]["hit_fraction"]) for c in CASES],
                     *[percentage(v) for v in means], f"{percentage(min(values))}-{percentage(max(values))}"])
    section("7. Spatial hit rates around true groups (%)", ["Offset", "n/run", "Intel C0", "Intel gather", "Intel C1", "AMD C0 mean", "AMD gather mean", "AMD C1 mean", "AMD gather range"], rows,
            "Offset unit=32 columns. Diagnostic uses truth; detection/filtering does not. Prefetch causality is untested.")

    rows = []
    for seed, r in ((42, primary), (43, records["prefill-seed43"])):
        for case, label in (("no_outliers", "No outliers"), ("injected_gather", "Gather")):
            numerics = r["summary"]["cases"][case]["numerics"]
            for key, method in (("normal_vs_reconstructed_weights", "Normal / reconstructed"),
                                ("mixed_vs_reconstructed_weights", "Mixed / reconstructed"),
                                ("mixed_vs_original_weights", "Mixed / original")):
                s = numerics[key]
                rows.append([seed, label, method, f"{s['max_absolute']:.6f}", f"{s['relative_l2']:.6f}"])
    section("8. Numerical checks", ["Seed", "Case", "Path / weight reference", "Max absolute error", "Relative L2"], rows,
            "Seed-42 metrics are identical in Intel and all AMD seed-42 datasets. These are arithmetic checks, not model accuracy.")

    section("9. Scope / evidence", ["Item", "Summary"], [
        ("Supported result", "All 10 matched AMD acquisitions recover the exact four-group set after filtering."),
        ("Comparison limit", "One historical Intel acquisition; original compiler/system state unavailable."),
        ("Coverage", "Synthetic, single-process CPU experiment; no real-LLM, NPU or cross-process result."),
        ("Generalization", "One input repeated 10 times; supplementary settings are not a one-factor ablation."),
        ("Filter limit", "May suppress genuinely adjacent true groups; these inputs have separated groups."),
        ("Validation", "405,840 raw records; 15 datasets; 17 pinned source hashes; raw files unchanged."),
        ("Source baseline", PINNED),
        ("Full details", "README.md; matched-figure4/REPORT.txt; per-run CSV/JSON and environment manifest."),
    ])
    return "\n".join(lines).rstrip()


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
    (BASE / "FINAL_RESULTS.txt").write_text(render_results(records) + "\n")
    print(json.dumps({"validated_datasets": len(records), "validated_raw_records": total,
                      "results_file": str(BASE / "FINAL_RESULTS.txt")}, indent=2))


if __name__ == "__main__":
    main()
