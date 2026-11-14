"""Acquire fresh prefill cache measurements for a range of observed-row counts.

python -m prefill_flush_reload.sweep --rows 1,2,4,8,16,32,64 --output /tmp/row-sweep

Each job runs in a separate process with its own calibration, controls, raw
samples, numerical checks, and guess reports. Jobs execute sequentially to
avoid competing for the cache. Their order is shuffled to reduce time drift.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


METRICS = ("precision", "recall", "f1", "accuracy", "jaccard")
SELECTORS = ("consensus", "control_adjusted")
PROCESSING = ("raw", "leading_group")


def integer_list(text):
    """Preserve user order while preventing accidental duplicate jobs."""
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Use a nonempty list without duplicate values")
    return values


def plan_jobs(args):
    """Use the same seeds and row starts at every count for paired comparisons."""
    jobs = []
    for count in args.rows:
        for seed in args.seeds:
            for start in args.row_starts:
                for run in range(1, args.runs + 1):
                    directory = args.output / f"rows-{count:04d}" / f"seed-{seed}" / f"start-{start:04d}" / f"run-{run:03d}"
                    jobs.append({"rows": count, "seed": seed, "row_start": start,
                                 "run": run, "directory": str(directory)})
    random.Random(args.order_seed).shuffle(jobs)
    return jobs


def child_command(args, job):
    """Forward all experimental settings explicitly, including flush scope."""
    options = {
        "tokens": args.tokens, "intermediate": args.intermediate, "hidden": args.hidden,
        "outliers": args.outliers, "threshold": args.threshold, "repeats": args.repeats,
        "probe-stride": args.probe_stride, "probe-mode": args.probe_mode,
        "flush-scope": args.flush_scope, "probe-rows": job["rows"], "row-start": job["row_start"],
        "seed": job["seed"], "row-hit-fraction": args.row_hit_fraction,
        "min-row-fraction": args.min_row_fraction, "min-control-delta": args.min_control_delta,
        "output": job["directory"],
    }
    return [sys.executable, "-m", "prefill_flush_reload",
            *[part for key, value in options.items() for part in (f"--{key}", str(value))]]


def collect_scores(job, exit_code):
    """A child timing FAIL (exit 1) is valid data, not an acquisition error."""
    if exit_code not in (0, 1):
        raise ValueError(f"Child setup/execution failed with exit status {exit_code}")
    folder = Path(job["directory"])
    analysis = json.loads((folder / "analysis.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if analysis["observed_rows"] != list(range(job["row_start"], job["row_start"] + job["rows"])):
        raise ValueError("Child observed-row range does not match its planned job")
    scores = []
    for selector in SELECTORS:
        for processing in PROCESSING:
            score = analysis["group_detection"]["methods"][selector][processing]
            scores.append({**job, "selector": selector, "processing": processing, **score,
                           "cache_check_passed": summary["cache_check_passed"]})
    return scores


def aggregate_scores(scores, counts, expected_per_count, target_precision, target_recall, target_rate):
    """Keep per-run variability and raw/processed scores separate.

    A success must meet BOTH precision and recall within the same acquisition.
    Failed jobs remain in the planned denominator rather than silently inflating
    the success rate. Empty predictions get zero macro precision/recall/F1.
    """
    aggregates = []
    for count in sorted(counts):
        for selector in SELECTORS:
            for processing in PROCESSING:
                subset = [score for score in scores if score["rows"] == count
                          and score["selector"] == selector and score["processing"] == processing]
                successes = sum(score["precision"] is not None and score["recall"] is not None
                                and score["precision"] >= target_precision and score["recall"] >= target_recall
                                for score in subset)
                result = {"rows": count, "selector": selector, "processing": processing,
                          "planned_runs": expected_per_count, "completed_runs": len(subset),
                          "successful_runs": successes, "success_rate": successes / expected_per_count,
                          "meets_target": len(subset) == expected_per_count and successes / expected_per_count >= target_rate,
                          "exact_match_rate": sum(score["exact_match"] for score in subset) / expected_per_count,
                          "cache_pass_rate": sum(score["cache_check_passed"] for score in subset) / expected_per_count}
                for metric in METRICS:
                    values = [score[metric] if score[metric] is not None else 0.0 for score in subset]
                    result[f"mean_{metric}"] = statistics.mean(values) if values else None
                    result[f"min_{metric}"] = min(values) if values else None
                    result[f"max_{metric}"] = max(values) if values else None
                for key in ("tp", "fp", "fn", "tn"):
                    result[f"total_{key}"] = sum(score[key] for score in subset)
                tp, fp, fn = (result[f"total_{key}"] for key in ("tp", "fp", "fn"))
                result["micro_precision"] = tp / (tp + fp) if tp + fp else None
                result["micro_recall"] = tp / (tp + fn) if tp + fn else None
                aggregates.append(result)
    recommendations = []
    for selector in SELECTORS:
        for processing in PROCESSING:
            passing = [result["rows"] for result in aggregates
                       if result["selector"] == selector and result["processing"] == processing and result["meets_target"]]
            recommendations.append({"selector": selector, "processing": processing,
                                    "minimum_tested_rows": min(passing) if passing else None,
                                    "passing_row_counts": passing})
    return aggregates, recommendations


def write_results(args, jobs, outcomes, scores):
    """Checkpoint machine-readable results and a row-count comparison table."""
    expected = len(args.seeds) * len(args.row_starts) * args.runs
    aggregates, recommendations = aggregate_scores(
        scores, args.rows, expected, args.target_precision, args.target_recall, args.target_success_rate)
    document = {"configuration": {**vars(args), "output": str(args.output)}, "jobs": jobs,
                "outcomes": outcomes, "aggregates": aggregates, "recommendations": recommendations,
                "resolution_columns": 32,
                "meaning": "Minimum tested rows meeting both precision and recall targets in the requested fraction of runs; not a universal sample requirement."}
    (args.output / "sweep.json").write_text(json.dumps(document, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (args.output / "runs.csv").open("w", newline="") as target:
        columns = ["rows", "seed", "row_start", "run", "selector", "processing", "directory",
                   "tp", "fp", "fn", "tn", *METRICS, "exact_match", "cache_check_passed",
                   "predicted_groups", "actual_groups", "false_positive_groups", "missed_groups"]
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for score in scores:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in score.items() if key in columns})
    fmt = lambda value: "n/a" if value is None else f"{value:.1%}"
    lines = ["# Observed-row sweep", "",
             f"Full prefill: {args.tokens} tokens. Probe mode: `{args.probe_mode}`; "
             f"stride: {args.probe_stride}; flush scope: `{args.flush_scope}`.", "",
             f"Seeds: {args.seeds}; row starts: {args.row_starts}; independent runs per combination: {args.runs}; "
             f"cache observations per run: {args.repeats} repeats. Completed jobs: {len(outcomes)}/{len(jobs)}.", "",
             "Each row count has fresh acquisition, calibration, controls, and full numerical checks. "
             "Only the observed-row count changes within a paired seed/start/run comparison. "
             "Jobs execute sequentially in shuffled order.", "",
             f"A successful run requires precision >= {args.target_precision:.0%} AND recall >= {args.target_recall:.0%}. "
             f"A row count meets the target when >= {args.target_success_rate:.0%} of planned runs succeed "
             "and every acquisition completes. Scores use 32-column groups.", "",
             "## First-group guesses", "",
             "| Rows | Selector | Completed | Mean precision | Mean recall | Mean F1 | F1 range | Exact-set rate | Target success | Meets target |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for result in aggregates:
        if result["processing"] != "leading_group":
            continue
        lines.append(f"| {result['rows']} | {result['selector']} | {result['completed_runs']}/{expected} | "
                     f"{fmt(result['mean_precision'])} | {fmt(result['mean_recall'])} | {fmt(result['mean_f1'])} | "
                     f"{fmt(result['min_f1'])}–{fmt(result['max_f1'])} | {fmt(result['exact_match_rate'])} | "
                     f"{fmt(result['success_rate'])} | {result['meets_target']} |")
    lines += ["", "## Smallest tested counts meeting the target", ""]
    for result in recommendations:
        if result["processing"] == "leading_group":
            value = result["minimum_tested_rows"]
            lines.append(f"- {result['selector']}: {value if value is not None else 'none yet'}; "
                         f"passing tested counts: {result['passing_row_counts']}.")
    failures = [outcome for outcome in outcomes if outcome["status"] == "error"]
    lines += ["", "## Interpretation and files", "",
              "These are empirical results for the selected seeds, row windows, and execution environment. "
              "The smallest passing count is not assumed to imply that every larger count passes. "
              "Use multiple seeds, row windows, and independent runs before judging reliability. "
              "Repeated runs with the same seed reuse synthetic data but perform fresh cache measurements.", "",
              "Raw candidate scores and leading-group scores are both in `comparison.csv`, "
              "individual runs in `runs.csv`, and all settings and job outcomes in `sweep.json`. "
              "Every run directory retains raw timings, per-row evidence, following candidate groups, "
              "numerical errors, and `execution.log`. A raw timing FAIL remains valid scoring data. "
              "Missing/failed acquisitions remain in target-success denominators. Undefined per-run "
              "precision/recall is counted as zero in macro means.", "",
              f"Acquisition errors: {len(failures)}."]
    for failure in failures:
        lines.append(f"- {failure['directory']}: {failure['error']}")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return recommendations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=integer_list, default=integer_list("1,2,4,8,16,32,64"))
    parser.add_argument("--seeds", type=integer_list, default=[42])
    parser.add_argument("--row-starts", type=integer_list, default=[0])
    parser.add_argument("--runs", type=int, default=1, help="independent acquisitions per count/seed/start")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=4864)
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--outliers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=32)
    parser.add_argument("--repeats", type=int, default=5, help="cache observations within each acquisition")
    parser.add_argument("--probe-mode", choices=("sweep", "single"), default="sweep")
    parser.add_argument("--probe-stride", type=int, default=1)
    parser.add_argument("--flush-scope", choices=("all", "observed"), default="observed")
    parser.add_argument("--row-hit-fraction", type=float, default=.6)
    parser.add_argument("--min-row-fraction", type=float, default=.5)
    parser.add_argument("--min-control-delta", type=float, default=.5)
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--target-recall", type=float, default=.95)
    parser.add_argument("--target-success-rate", type=float, default=.9)
    parser.add_argument("--order-seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=300, help="seconds allowed per child acquisition")
    parser.add_argument("--output", type=Path, default=Path("/tmp/prefill-row-sweep"))
    args = parser.parse_args()
    fractions = (args.row_hit_fraction, args.min_row_fraction, args.min_control_delta,
                 args.target_precision, args.target_recall, args.target_success_rate)
    if not all(math.isfinite(value) and 0 < value <= 1 for value in fractions):
        parser.error("Detection and target fractions must be in (0, 1]")
    if args.tokens < 2 or args.runs < 1 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Require tokens>=2, runs/repeats>=1, and a positive finite timeout")
    if any(count < 1 for count in args.rows) or any(start < 0 or start + max(args.rows) > args.tokens for start in args.row_starts):
        parser.error("Every requested row count/start must fit inside the full prefill")
    if (args.intermediate < 64 or args.intermediate % 32 or args.hidden < 1
            or not 1 <= args.outliers < args.intermediate or not 1 < args.threshold <= 100
            or not 1 <= args.probe_stride <= args.intermediate // 32
            or args.intermediate * 127**2 > 2**31 - 1):
        parser.error("Invalid layer dimensions, outlier settings, threshold, or probe stride")
    args.output = args.output.resolve()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be a new or empty directory to preserve previous measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(args)
    outcomes, scores = [], []
    write_results(args, jobs, outcomes, scores)
    root = Path(__file__).resolve().parent.parent
    for index, job in enumerate(jobs, 1):
        folder = Path(job["directory"])
        folder.mkdir(parents=True, exist_ok=True)
        command = child_command(args, job)
        (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"[{index}/{len(jobs)}] rows={job['rows']}, seed={job['seed']}, "
              f"start={job['row_start']}, run={job['run']}", flush=True)
        started = time.monotonic()
        outcome = {**job}
        try:
            with (folder / "execution.log").open("w") as log:
                child = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False)
            collected = collect_scores(job, child.returncode)
            scores.extend(collected)
            outcome.update(status="complete", exit_code=child.returncode)
            leading = next(score for score in collected if score["selector"] == "consensus"
                           and score["processing"] == "leading_group")
            print(f"  Leading guesses: TP={leading['tp']}, FP={leading['fp']}, FN={leading['fn']}; "
                  f"exact={leading['exact_match']}", flush=True)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            outcome.update(status="error", error=str(error))
            print(f"  Acquisition error: {error}; see {folder / 'execution.log'}", flush=True)
        outcome["elapsed_seconds"] = time.monotonic() - started
        outcomes.append(outcome)
        recommendations = write_results(args, jobs, outcomes, scores)
    for result in recommendations:
        if result["processing"] == "leading_group":
            print(f"{result['selector']}: smallest tested count meeting target = "
                  f"{result['minimum_tested_rows']}", flush=True)
    print(f"Comparison report: {args.output / 'report.md'}", flush=True)
    return 2 if any(outcome["status"] == "error" for outcome in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
