"""Analyze new or existing lines.csv without rerunning the native experiment.

python -m prefill_flush_reload.analyze RESULTS --probe-rows 4 --row-start 0

Row filtering here only parses an existing scan. It does NOT retroactively
remove interference from unselected rows that were probed during acquisition.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from .scoring import detection_scores


# Keep statistical filtering independent of injected labels. Ground truth is
# retained solely for evaluating results and diagnosing their spatial footprint.
def analyze_csv(path, row_start=0, probe_rows=None, row_hit_fraction=.6, min_row_fraction=.5,
                min_control_delta=.5):
    if row_start < 0 or (probe_rows is not None and probe_rows < 1):
        raise ValueError("Row start must be nonnegative and row count positive")
    if not all(math.isfinite(value) and 0 < value <= 1
               for value in (row_hit_fraction, min_row_fraction, min_control_delta)):
        raise ValueError("Detection fractions must be in (0, 1]")
    cells = defaultdict(lambda: [0, 0])
    truth = set()
    probe_records = []
    # Cells aggregate repeat counts per case, row, and 32-column section.
    with Path(path).open(newline="") as source:
        for record in csv.DictReader(source):
            row = int(record["row"])
            if row < row_start or (probe_rows is not None and row >= row_start + probe_rows):
                continue
            case, start = record["case"], int(record["column_start"])
            cell = cells[case, row, start]
            cell[0] += int(record["predicted_hit"])
            cell[1] += 1
            if record.get("probe_index", "") != "":
                probe_records.append({"case": case, "trial": int(record["trial"]), "row": row,
                                      "start": start, "index": int(record["probe_index"]),
                                      "previous": (int(record["previous_probe_line"])
                                                   if record.get("previous_probe_line", "") != "" else None),
                                      "hit": int(record["predicted_hit"]),
                                      "expected": int(record["expected_access"])})
            if case == "injected_gather" and int(record["outlier_line"]):
                truth.add((row, start))
    if not cells:
        raise ValueError("No measurements exist for the selected rows")
    cases = sorted({case for case, _, _ in cells})
    rows = sorted({row for _, row, _ in cells})
    starts = sorted({start for _, _, start in cells})
    groups = []
    for start in starts:
        details = []
        for row in rows:
            counts = {}
            for case in cases:
                hits, total = cells.get((case, row, start), (0, 0))
                if total:
                    counts[case] = {"hits": hits, "observations": total, "hit_fraction": hits / total}
            positive = counts.get("injected_gather")
            controls = [counts.get(name) for name in ("no_outliers", "injected_no_gather")]
            baseline = max(control["hit_fraction"] for control in controls) if all(controls) else None
            fraction = positive["hit_fraction"] if positive else None
            delta = fraction - baseline if fraction is not None and baseline is not None else None
            details.append({"row": row, "cases": counts,
                            "any_hit": bool(positive and positive["hits"]),
                            "consistent_hit": fraction is not None and fraction >= row_hit_fraction,
                            "control_delta": delta,
                            "control_adjusted_hit": (fraction is not None and fraction >= row_hit_fraction
                                                     and delta is not None and delta >= min_control_delta)})
        any_rows = [item["row"] for item in details if item["any_hit"]]
        consistent = [item["row"] for item in details if item["consistent_hit"]]
        adjusted = [item["row"] for item in details if item["control_adjusted_hit"]]
        measured_rows = [item["row"] for item in details if "injected_gather" in item["cases"]]
        groups.append({"column_start": start, "column_end_exclusive": start + 32,
                       "contains_outlier": any((row, start) in truth for row in rows),
                       "any_hit_rows": any_rows, "any_hit_row_count": len(any_rows),
                       "consistent_hit_rows": consistent, "consistent_hit_row_count": len(consistent),
                       "control_adjusted_rows": adjusted, "control_adjusted_row_count": len(adjusted),
                       "detected": bool(measured_rows) and len(consistent) / len(measured_rows) >= min_row_fraction,
                       "control_adjusted_detected": bool(measured_rows) and len(adjusted) / len(measured_rows) >= min_row_fraction,
                       "rows": details})

    # Estimate a halo around KNOWN injected sections. This diagnostic uses truth;
    # it is not used by the detection rules or to erase inconvenient false hits.
    spatial = []
    for offset in range(-4, 9):
        row = {"offset_groups": offset, "offset_columns": offset * 32}
        for case in cases:
            keys = {(r, start + offset * 32) for r, start in truth
                    if (offset == 0 or (r, start + offset * 32) not in truth)
                    and (case, r, start + offset * 32) in cells}
            hits = sum(cells[case, r, start][0] for r, start in keys)
            total = sum(cells[case, r, start][1] for r, start in keys)
            row[case] = {"hits": hits, "observations": total,
                         "hit_fraction": hits / total if total else None}
        spatial.append(row)

    # Contrast first probes with later probes near a previous reload. These
    # observational associations do not by themselves identify a prefetcher.
    temporal_counts = defaultdict(lambda: [0, 0])
    lines_per_row = max(starts) // 32 + 1
    lookup = {(item["case"], item["trial"], item["row"] * lines_per_row + item["start"] // 32): item
              for item in probe_records}
    for item in probe_records:
        if item["expected"]:
            continue
        if item["index"] == 0:
            bucket = "first_probe_in_window"
        else:
            previous = lookup.get((item["case"], item["trial"], item["previous"]))
            if previous is None:
                continue
            if previous["hit"]:
                near = (previous["row"] == item["row"]
                        and abs(previous["start"] - item["start"]) <= 4 * 32)
                bucket = "after_nearby_hit" if near else "after_distant_hit"
            else:
                bucket = "after_miss"
        counter = temporal_counts[item["case"], bucket]
        counter[0] += item["hit"]
        counter[1] += 1
    temporal = [{"case": case, "condition": bucket, "false_hits": hits, "observations": total,
                 "false_positive_rate": hits / total}
                for (case, bucket), (hits, total) in sorted(temporal_counts.items())]

    # A contiguous run is ambiguous: expose its extent rather than assuming the
    # first group is the true feature and deleting potentially real neighbors.
    clusters = []
    for group in groups:
        if not group["detected"]:
            continue
        if clusters and clusters[-1]["column_end_exclusive"] == group["column_start"]:
            clusters[-1]["column_end_exclusive"] = group["column_end_exclusive"]
            clusters[-1]["group_count"] += 1
        else:
            clusters.append({"column_start": group["column_start"],
                             "column_end_exclusive": group["column_end_exclusive"], "group_count": 1})
    scored = detection_scores(groups)
    return {"observed_rows": rows, "row_count": len(rows),
            "guessed_groups": scored["methods"]["consensus"]["leading_group"]["predicted_groups"],
            "control_adjusted_guessed_groups": scored["methods"]["control_adjusted"]["leading_group"]["predicted_groups"],
            "rules": {"row_hit_fraction": row_hit_fraction, "min_row_fraction": min_row_fraction,
                      "min_control_delta": min_control_delta},
            "detected_groups": [group["column_start"] for group in groups if group["detected"]],
            "control_adjusted_groups": [group["column_start"] for group in groups if group["control_adjusted_detected"]],
            "ambiguous_contiguous_clusters": clusters, "group_detection": scored,
            "groups": groups, "spatial_offsets": spatial,
            "probe_order_diagnostic": temporal}


def row_ranges(rows):
    """Compact row identities without losing which rows were detected."""
    runs = []
    for row in rows:
        if runs and row == runs[-1][1] + 1:
            runs[-1][1] = row
        else:
            runs.append([row, row])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs) or "none"


def write_analysis(analysis, output):
    """Write a concise human report plus exhaustive per-row machine reports."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    with (output / "row_sections.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("column_start", "column_end_exclusive", "row", "case", "hits", "observations",
                         "hit_fraction", "gather_consistent_hit", "gather_control_delta", "gather_control_adjusted_hit"))
        for group in analysis["groups"]:
            for row in group["rows"]:
                for case, counts in row["cases"].items():
                    writer.writerow((group["column_start"], group["column_end_exclusive"], row["row"], case,
                                     counts["hits"], counts["observations"], counts["hit_fraction"],
                                     int(row["consistent_hit"]), row["control_delta"], int(row["control_adjusted_hit"])))
    # Flat scores support later experiments varying observed-row counts without
    # parsing Markdown. Raw and heuristic scores always remain distinguishable.
    with (output / "group_scores.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("row_count", "observed_rows", "row_hit_fraction", "min_row_fraction", "min_control_delta",
                         "selector", "processing", "tp", "fp", "fn", "tn",
                         "precision", "recall", "f1", "jaccard", "accuracy", "exact_match",
                         "predicted_groups", "actual_groups"))
        for selector, method in analysis["group_detection"]["methods"].items():
            for processing in ("raw", "leading_group"):
                score = method[processing]
                writer.writerow((analysis["row_count"], json.dumps(analysis["observed_rows"]),
                                 *[analysis["rules"][key] for key in
                                   ("row_hit_fraction", "min_row_fraction", "min_control_delta")], selector, processing,
                                 *[score[key] for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1",
                                                        "jaccard", "accuracy", "exact_match")],
                                 json.dumps(score["predicted_groups"]), json.dumps(score["actual_groups"])))
    rules = analysis["rules"]
    lines = ["# Cache-line detection report", "",
             f"Observed token rows ({analysis['row_count']}): {row_ranges(analysis['observed_rows'])}.", "",
             "Ranges below are 32-column cache-line groups, with an exclusive upper bound. "
             "They do not identify an exact feature column.", "",
             f"A row is consistent when >= {rules['row_hit_fraction']:.0%} of its repeats hit. "
             f"A group is detected when >= {rules['min_row_fraction']:.0%} of observed rows are consistent. "
             f"Control-adjusted rows additionally exceed both controls by >= {rules['min_control_delta']:.0%}.", "",
             "## Guessed groups: first candidate in each consecutive run", "",
             "The primary guesses below keep only the lowest-column group of each consecutive "
             "candidate run. Following candidates remain listed as possible prefetch hits. "
             "A run is formed by exact 32-column adjacency, without bridging gaps. "
             "This heuristic can suppress genuinely adjacent outlier groups.", "",
             "Scores compare 32-column groups, not exact individual feature columns. "
             "Multiple real columns in one group count as one actual group. "
             "Precision/recall/F1 and exact-set match are more useful than overall accuracy "
             "when true groups are sparse.", "",
             "| Selector | Processing | TP | FP | FN | Precision | Recall | F1 | Accuracy | Exact set |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for selector, method in analysis["group_detection"]["methods"].items():
        for processing in ("raw", "leading_group"):
            score = method[processing]
            metric = lambda value: "n/a" if value is None else f"{value:.2%}"
            lines.append(f"| {selector} | {processing} | {score['tp']} | {score['fp']} | {score['fn']} | "
                         f"{metric(score['precision'])} | {metric(score['recall'])} | {metric(score['f1'])} | "
                         f"{metric(score['accuracy'])} | {score['exact_match']} |")
    lines += ["", "| Selector | Guessed columns | Following candidate groups (starts) | Actual guess? | Real groups suppressed |",
              "|---|---|---|---|---|"]
    for selector, method in analysis["group_detection"]["methods"].items():
        for run in method["candidate_runs"]:
            start = run["guessed_group"]
            lines.append(f"| {selector} | {start}:{start + 32} | {run['following_groups']} | "
                         f"{run['guessed_group_is_actual']} | {run['following_actual_groups']} |")
    lines += ["", "## Raw candidate groups and row identities", "",
             "The table includes consensus detections and injected groups, including any missed groups. "
             "Exact per-row hit counts, all one-off hits, and all undetected groups appear in "
             "`row_sections.csv` and `analysis.json`.", "",
             "| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |",
             "|---|---|---|---|---|---|"]
    for group in analysis["groups"]:
        if not group["detected"] and not group["contains_outlier"]:
            continue
        entries = [f"{group['column_start']}:{group['column_end_exclusive']}",
                   "yes" if group["contains_outlier"] else "no"]
        for key in ("any_hit_rows", "consistent_hit_rows", "control_adjusted_rows"):
            entries.append(f"{len(group[key])}; {row_ranges(group[key])}")
        entries.append(f"{group['detected']} / {group['control_adjusted_detected']}")
        lines.append("| " + " | ".join(entries) + " |")
    lines += ["", "## Spatial halo diagnostic", "",
              "Hit rates around known injected groups; other true outlier groups are excluded "
              "at nonzero offsets. Positive offsets mean higher column indices in the SAME row, "
              "not later probes. These labels evaluate the footprint and do not filter detections.", "",
              "| Offset (groups) | No outliers | Gather | Skipped gather |", "|---|---|---|---|"]
    for entry in analysis["spatial_offsets"]:
        values = []
        for case in ("no_outliers", "injected_gather", "injected_no_gather"):
            fraction = entry.get(case, {}).get("hit_fraction")
            values.append("n/a" if fraction is None else f"{fraction:.2%}")
        lines.append("| " + " | ".join([str(entry["offset_groups"]), *values]) + " |")
    lines += ["", "## Probe-order diagnostic", "",
              "Non-accessed lines only. Nearby means within four groups in the same row. "
              "This association is not proof of causation; old CSVs have no probe-order metadata.", "",
              "| Case | Previous probe condition | False-hit rate | Observations |", "|---|---|---|---|"]
    for entry in analysis["probe_order_diagnostic"]:
        lines.append(f"| {entry['case']} | {entry['condition']} | "
                     f"{entry['false_positive_rate']:.2%} | {entry['observations']} |")
    lines += ["", "## Interpretation", "",
              "Consensus can reject occasional noise. Control adjustment can reject background "
              "hits shared with the no-gather controls. Neither removes a repeatable halo caused "
              "by the gather itself. The leading-group guesses apply a forward-halo assumption; "
              "raw detections, following candidates, and any real groups suppressed by that "
              "assumption remain visible. Ground truth evaluates guesses but never selects them.", "",
              "A row subset parsed from an old CSV still contains interference from the original "
              "full scan. Use an actual row-limited run to change the measurement protocol.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--probe-rows", type=int)
    parser.add_argument("--row-hit-fraction", type=float, default=.6)
    parser.add_argument("--min-row-fraction", type=float, default=.5)
    parser.add_argument("--min-control-delta", type=float, default=.5)
    args = parser.parse_args()
    try:
        analysis = analyze_csv(args.results / "lines.csv", args.row_start, args.probe_rows,
                               args.row_hit_fraction, args.min_row_fraction, args.min_control_delta)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    destination = args.output or args.results
    write_analysis(analysis, destination)
    print(f"Analyzed rows {row_ranges(analysis['observed_rows'])}; "
          f"detected group starts: {analysis['detected_groups']}")
    print(f"Control-adjusted group starts: {analysis['control_adjusted_groups']}")
    for selector, method in analysis["group_detection"]["methods"].items():
        score = method["leading_group"]
        print(f"{selector} leading guesses: {score['predicted_groups']}; "
              f"TP={score['tp']}, FP={score['fp']}, FN={score['fn']}, exact={score['exact_match']}")
    print(f"Report: {destination / 'report.md'}")


if __name__ == "__main__":
    main()
