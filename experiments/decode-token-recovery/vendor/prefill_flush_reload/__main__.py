"""CLI orchestration for a synthetic prefill down-projection experiment."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .analyze import analyze_csv, write_analysis
from .measurement import calibrate, observe, summarize
from .native import Native
from .simulation import Simulation


def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64, help="prefill token rows (at least 2)")
    parser.add_argument("--intermediate", type=int, default=4864, help="input features; multiple of 32")
    parser.add_argument("--hidden", type=int, default=896, help="output features")
    parser.add_argument("--outliers", type=int, default=4, help="injected feature columns")
    parser.add_argument("--threshold", type=float, default=32, help="activation magnitude threshold")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probe-stride", type=int, default=4, help="interleaved fresh sweeps per trial")
    parser.add_argument("--probe-rows", type=int, help="reload this many rows; full prefill still runs")
    parser.add_argument("--row-start", type=int, default=0, help="first token row to reload")
    parser.add_argument("--probe-mode", choices=("sweep", "single"), default="sweep",
                        help="single gives each target line its own fresh flush/gather")
    parser.add_argument("--flush-scope", choices=("all", "observed"), default="all",
                        help="flush full source or only the observed rows")
    parser.add_argument("--row-hit-fraction", type=float, default=.6,
                        help="repeat hit fraction required to mark a row consistent")
    parser.add_argument("--min-row-fraction", type=float, default=.5,
                        help="fraction of consistent rows required to detect a group")
    parser.add_argument("--min-control-delta", type=float, default=.5,
                        help="extra hit fraction over both controls for adjusted detections")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/tmp/prefill-flush-reload"))
    args = parser.parse_args()
    if (args.tokens < 2 or args.intermediate < 64 or args.intermediate % 32 or args.hidden < 1
            or not 1 <= args.outliers < args.intermediate or args.repeats < 1 or args.probe_stride < 1 or args.probe_stride > args.intermediate // 32
            or not 1 < args.threshold <= 100):
        parser.error("Require tokens>=2, intermediate>=64 divisible by 32, hidden>=1, "
                     "1<=outliers<intermediate, repeats>=1, 1<=probe-stride<=intermediate/32, "
                     "and 1<threshold<=100")
    if args.probe_rows is None:
        args.probe_rows = args.tokens - args.row_start
    if not 0 <= args.row_start < args.tokens or not 1 <= args.probe_rows <= args.tokens - args.row_start:
        parser.error("Observed rows must lie inside the full prefill matrix")
    if not all(0 < value <= 1 for value in
               (args.row_hit_fraction, args.min_row_fraction, args.min_control_delta)):
        parser.error("Detection fractions must be in (0, 1]")
    # Bound the scalar INT32 dot product regardless of injected magnitudes.
    if args.intermediate * 127**2 > 2**31 - 1:
        parser.error("intermediate dimension exceeds this kernel's safe INT32 accumulation bound")
    args.output.mkdir(parents=True, exist_ok=True)
    native = Native()
    simulation = None
    try:
        simulation = Simulation(native, args.tokens, args.intermediate, args.hidden,
                                args.seed, args.threshold, args.outliers)
        cutoff, calibration = calibrate(native, args.seed)
        print(f"Prefill A=[{args.tokens}, {args.intermediate}], "
              f"W=[{args.hidden}, {args.intermediate}], activation threshold={args.threshold}", flush=True)
        print(f"Reload rows={args.row_start}:{args.row_start + args.probe_rows}; "
              f"mode={args.probe_mode}; flush={args.flush_scope}; full gather remains unchanged", flush=True)
        print(f"Reload threshold={cutoff} TSC ticks; held-out primitive balanced accuracy="
              f"{calibration['balanced_accuracy']:.2%}", flush=True)
        report = {"configuration": {**vars(args), "output": str(args.output)},
                  "storage": "FP16 original activations and weights; INT8 quantized tensors; FP32 scales",
                  "arithmetic": "INT32 main accumulation; FP32 correction accumulation; FP16 output rounding",
                  "calibration": {"reload_threshold_tsc_ticks": cutoff, **calibration}, "probe_protocol": ("fresh flush/gather per target line" if args.probe_mode == "single" else
                                     "interleaved fresh flush/gather sweeps; reconstructed footprint, not one snapshot"),
                  "cases": {}}
        with (args.output / "lines.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("case", "trial", "row", "column_start", "column_end_exclusive",
                             "latency_tsc_ticks", "predicted_hit", "expected_access", "outlier_line",
                             "sweep", "probe_index", "previous_probe_line"))
            # The skipped-gather control retains the same injected outliers and
            # quantization, but deliberately does not execute their FP16 reads.
            for case, injected, gather in (("no_outliers", False, True),
                                            ("injected_gather", True, True),
                                            ("injected_no_gather", True, False)):
                columns = simulation.prepare(injected)
                outlier_lines = simulation.touched_lines(columns)
                accessed = outlier_lines if gather else set()
                trace = []
                timings = observe(simulation, gather, args.repeats, args.seed + 19, args.probe_stride,
                                  args.probe_rows, args.row_start, args.probe_mode, args.flush_scope, trace)
                order_metadata = {(trial, line): (sweep, index, previous)
                                  for trial, line, sweep, index, previous in trace}
                result = summarize(simulation, timings, accessed, outlier_lines, cutoff)
                result["outlier_columns"] = columns
                result["outlier_count"] = len(columns)
                for trial, latencies in enumerate(timings):
                    for line, latency in latencies.items():
                        row, section = divmod(line, args.intermediate // 32)
                        writer.writerow((case, trial, row, section * 32, (section + 1) * 32,
                                         latency, int(latency < cutoff), int(line in accessed),
                                         int(line in outlier_lines), *order_metadata[trial, line]))
                # Numerical checks occur only after all reloads, using the
                # gathered copy left by the final trial. Skip the intentional
                # omitted-correction control rather than using stale subA.
                if gather:
                    result["numerics"] = simulation.numerical_check()
                report["cases"][case] = result
                print(f"{case}: outliers={len(columns)}, precision={percent(result['precision'])}, "
                      f"recall={percent(result['recall'])}, FPR={percent(result['false_positive_rate'])}",
                      flush=True)
                if gather:
                    errors = result["numerics"]
                    print("  Relative L2 vs reconstructed-weight reference: "
                          f"normal={errors['normal_vs_reconstructed_weights']['relative_l2']:.6g}, "
                          f"mixed={errors['mixed_vs_reconstructed_weights']['relative_l2']:.6g}; "
                          "independent scalar checks passed", flush=True)
        analysis = analyze_csv(args.output / "lines.csv", row_hit_fraction=args.row_hit_fraction,
                               min_row_fraction=args.min_row_fraction,
                               min_control_delta=args.min_control_delta)
        write_analysis(analysis, args.output)
        report["detection_summary"] = {
            "observed_rows": analysis["observed_rows"], "detected_groups": analysis["detected_groups"],
            "control_adjusted_groups": analysis["control_adjusted_groups"], "rules": analysis["rules"],
            "group_detection": analysis["group_detection"],
            "details": "analysis.json", "human_report": "report.md", "scores": "group_scores.csv"}
        print(f"Detected 32-column group starts: {analysis['detected_groups']}", flush=True)
        print(f"Control-adjusted group starts: {analysis['control_adjusted_groups']}", flush=True)
        for selector, method in analysis["group_detection"]["methods"].items():
            score = method["leading_group"]
            print(f"{selector} leading guesses: {score['predicted_groups']}; "
                  f"precision={percent(score['precision'])}, recall={percent(score['recall'])}, "
                  f"F1={percent(score['f1'])}, exact={score['exact_match']}", flush=True)
        # Keep numerical correctness separate from environment-sensitive timing.
        measured = report["cases"]["injected_gather"]
        controls = [report["cases"][name]["false_positive_rate"]
                    for name in ("no_outliers", "injected_no_gather")]
        reliable = (calibration["balanced_accuracy"] >= .90
                    and measured["recall"] is not None and measured["recall"] >= .90
                    and measured["false_positive_rate"] is not None
                    and measured["false_positive_rate"] <= .10 and max(controls) <= .10)
        report["cache_criteria"] = "calibration>=90%, gather recall>=90%, all case FPR<=10%"
        report["cache_check_passed"] = reliable
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Cache check: {'PASS' if reliable else 'FAIL'}; artifacts: {args.output}", flush=True)
        if not reliable:
            print("Inspect distributions and section hit fractions; a timing failure does not "
                  "invalidate the numerical path. Larger matrices can evict gathered lines.")
        return 0 if reliable else 1
    finally:
        if simulation is not None:
            simulation.close()
        native.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        sys.exit(2)
