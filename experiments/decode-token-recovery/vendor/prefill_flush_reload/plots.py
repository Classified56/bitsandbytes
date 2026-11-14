"""Export publication figures from existing results, without new measurements.

python -m prefill_flush_reload.plots prefill_flush_reload --output /tmp/prefill-paper

Requires Matplotlib. SVG/PDF preserve vector elements; PNG defaults to 600 dpi.
All group positions use original feature-column coordinates, not packed indices.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

from .plot_data import compatible_cohorts, discover_results, load_result


COLORS = {"injected": "#D55E00", "raw": "#0072B2", "guess": "#009E73"}


def short_settings(record):
    meta = record["metadata"]
    return (f"{record['row_count']} rows, seed {meta['seed']}, start {record['rows'][0]}; "
            f"{meta['probe_mode']}, stride {meta['probe_stride']}, flush {meta['flush_scope']}")


def save_figure(figure, destination, formats, dpi):
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = destination.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    return paths


def group_strips(axes, record, selector):
    """Show injected groups, unfiltered candidates, and heuristic guesses."""
    from matplotlib.patches import Rectangle
    method = record["scores"]["methods"][selector]
    series = [("Outlier Column", record["scores"]["actual_groups"], COLORS["injected"]),
              ("Raw candidates", method["raw"]["predicted_groups"], COLORS["raw"]),
              ("First-group guesses", method["leading_group"]["predicted_groups"], COLORS["guess"])]
    for axis, (label, starts, color) in zip(axes, series):
        positions = set(starts)
        axis.set_facecolor("#F4F4F4")
        for start in sorted(positions):
            axis.add_patch(Rectangle((start, -.5), 32, 1, facecolor=color, edgecolor="none"))
        axis.set_ylim(-.5, .5)
        axis.set_yticks([0], [label], fontsize=8)
        axis.tick_params(axis="y", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlim(0, record["width"])
    # Exact injected feature coordinates supplement group-level truth. Their
    # marker shape and caption distinguish them from recoverable cache groups.
    axes[0].scatter(record["injected_columns"], [0] * len(record["injected_columns"]),
                    marker="|", color="black", s=55, linewidths=.8, zorder=3)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False, bottom=False)


def plot_run(record, selector, folder, plt, formats, dpi, width):
    """Separate sparse-location overview and row-by-group evidence figure."""
    paths = []
    figure, axes = plt.subplots(3, 1, figsize=(width, 2.25), sharex=True,
                               gridspec_kw={"hspace": .20})
    group_strips(axes, record, selector)
    figure.suptitle(record["label"], fontsize=10, y=1.06)
    axes[0].set_title(short_settings(record), fontsize=8, pad=8)
    axes[-1].set_xlabel("FFN intermediate feature column (32 columns per group)")
    paths += save_figure(figure, folder / "groups", formats, dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(width, max(2.3, min(5, 1.7 + record["row_count"] * .045))))
    image = axis.imshow(record["hit_fractions"], origin="upper", aspect="auto", interpolation="nearest",
                        cmap="viridis", vmin=0, vmax=1,
                        extent=(0, record["width"], record["row_count"] - .5, -.5))
    for start in record["scores"]["actual_groups"]:
        axis.axvspan(start, start + 32, facecolor="none", edgecolor=COLORS["injected"], linewidth=.9)
    rows = record["rows"]
    step = max(1, len(rows) // 8)
    positions = sorted(set([*range(0, len(rows), step), len(rows) - 1]))
    axis.set_yticks(positions, [rows[index] for index in positions])
    axis.set_ylabel("Observed token row")
    axis.set_xlabel("FFN intermediate feature column")
    axis.set_title(f"{record['label']}: reload hit fraction\n{short_settings(record)}", fontsize=9)
    figure.colorbar(image, ax=axis, label="Hit fraction across repeats", fraction=.035, pad=.025)
    paths += save_figure(figure, folder / "rows", formats, dpi)
    plt.close(figure)
    return paths


def plot_overview(records, selector, output, plt, formats, dpi, width):
    """A location comparison labels each acquisition, including confounders."""
    paths = []
    by_width = {}
    for record in records:
        by_width.setdefault(record["width"], []).append(record)
    index = 0
    for _, members in sorted(by_width.items()):
        members.sort(key=lambda item: (item["row_count"], item["source"]))
        for start in range(0, len(members), 8):
            batch = members[start:start + 8]
            index += 1
            figure, axes = plt.subplots(3 * len(batch), 1, figsize=(width, 1 + len(batch) * 1.25),
                                       sharex=True, gridspec_kw={"hspace": .55}, squeeze=False)
            for number, record in enumerate(batch):
                strip = [axes[3 * number + offset, 0] for offset in range(3)]
                group_strips(strip, record, selector)
                strip[0].set_title(f"{record['label']} — {short_settings(record)}", fontsize=7, loc="left", pad=3)
                if number < len(batch) - 1:
                    strip[-1].tick_params(axis="x", bottom=False, labelbottom=False)
            axes[-1, 0].set_xlabel("FFN intermediate feature column (group width = 32)")
            paths += save_figure(figure, output / f"overview-{index:02d}", formats, dpi)
            plt.close(figure)
    return paths


def plot_cohort(records, selector, destination, plt, formats, dpi, width):
    """Connect only comparable acquisitions; ranges are descriptive, not CIs."""
    by_count = {}
    for record in records:
        by_count.setdefault(record["row_count"], []).append(record)
    if len(by_count) < 2:
        return []
    figure, axes = plt.subplots(1, 3, figsize=(width, 2.5), sharey=True)
    counts = sorted(by_count)
    for axis, metric in zip(axes, ("precision", "recall", "f1")):
        for processing, label, color, marker in (("raw", "Raw candidates", COLORS["raw"], "o"),
                                                 ("leading_group", "First-group guesses", COLORS["guess"], "s")):
            values = [[record["scores"]["methods"][selector][processing][metric]
                       for record in by_count[count]] for count in counts]
            values = [[0 if value is None else value for value in group] for group in values]
            means = [statistics.mean(group) for group in values]
            axis.plot(counts, means, marker=marker, color=color, label=label, linewidth=1.1, markersize=4,
                      linestyle="--" if processing == "raw" else "-")
            # Only show a min/max range where multiple acquisitions exist.
            for count, group, mean in zip(counts, values, means):
                if len(group) > 1:
                    axis.errorbar([count], [mean], yerr=[[mean - min(group)], [max(group) - mean]],
                                  color=color, capsize=2, linewidth=.7)
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts, [str(count) for count in counts])
        axis.set_xlabel("Observed token rows")
        axis.set_title(metric.upper() if metric == "f1" else metric.capitalize())
        axis.set_ylim(-.03, 1.03)
        axis.grid(axis="y", alpha=.25, linewidth=.5)
    axes[0].set_ylabel("Group-level score")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    meta = records[0]["metadata"]
    figure.suptitle(f"Seed {meta['seed']}; start {meta['row_start']}; {meta['probe_mode']}; "
                   f"stride {meta['probe_stride']}; flush {meta['flush_scope']}", fontsize=8, y=1.05)
    paths = save_figure(figure, destination, formats, dpi)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="result directories or trees of results")
    parser.add_argument("--output", type=Path, default=Path("research_figures/prefill"))
    parser.add_argument("--selector", choices=("consensus", "control_adjusted"), default="consensus")
    parser.add_argument("--formats", default="pdf,svg,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    args = parser.parse_args()
    formats = args.formats.split(",")
    if not formats or any(extension not in ("pdf", "svg", "png") for extension in formats):
        parser.error("Formats must be a comma-separated subset of pdf,svg,png")
    if args.dpi < 72 or not 3 <= args.width <= 20:
        parser.error("Require dpi>=72 and width between 3 and 20 inches")
    try:
        folders = discover_results(args.inputs, excluded=args.output)
    except ValueError as error:
        parser.error(str(error))
    if not folders:
        parser.error("No measurement summary.json files were found")
    # A writable, isolated cache allows headless rendering on restricted hosts.
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("Matplotlib is required: python -m pip install matplotlib")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    records, excluded = [], []
    for folder in folders:
        try:
            records.append(load_result(folder))
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append({"source": str(folder), "error": str(error)})
            print(f"Excluded {folder}: {error}", file=sys.stderr)
    if not records:
        parser.error("No valid result folders; see the exclusions above")
    artifacts = plot_overview(records, args.selector, args.output, plt, formats, args.dpi, args.width)
    manifest = {"selector": args.selector, "matplotlib_version": matplotlib.__version__,
                "group_width": 32, "dpi": args.dpi, "width_inches": args.width,
                "excluded": excluded, "sources": [], "cohorts": [], "artifacts": artifacts}
    with (args.output / "plotted_groups.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("source", "row_count", "seed", "group_start", "group_end_exclusive",
                         "injected", "raw_candidate", "leading_guess", "mean_hit_fraction"))
        for record in records:
            identity = hashlib.sha256(record["source"].encode()).hexdigest()[:8]
            folder = args.output / "runs" / f"{record['label']}-{identity}"
            artifacts += plot_run(record, args.selector, folder, plt, formats, args.dpi, args.width)
            method = record["scores"]["methods"][args.selector]
            actual, raw, guessed = (set(record["scores"]["actual_groups"]),
                                    set(method["raw"]["predicted_groups"]),
                                    set(method["leading_group"]["predicted_groups"]))
            for index, group in enumerate(record["groups"]):
                start = group["column_start"]
                writer.writerow((record["source"], record["row_count"], record["metadata"]["seed"], start, start + 32,
                                 int(start in actual), int(start in raw), int(start in guessed),
                                 statistics.mean(row[index] for row in record["hit_fractions"])))
            with (folder / "hit_fractions.csv").open("w", newline="") as heatmap:
                csv.writer(heatmap).writerows([["token_row", *range(0, record["width"], 32)],
                                              *[[row, *values] for row, values in zip(record["rows"], record["hit_fractions"])]])
            manifest["sources"].append({key: record[key] for key in
                                        ("source", "label", "rows", "metadata", "analysis_source", "injected_columns", "scores")})
    for index, members in enumerate(compatible_cohorts(records), 1):
        generated = plot_cohort(members, args.selector, args.output / f"accuracy-cohort-{index:02d}",
                                plt, formats, args.dpi, args.width)
        artifacts += generated
        manifest["cohorts"].append({"id": index, "settings": members[0]["metadata"],
                                    "sources": [record["source"] for record in members],
                                    "observed_row_counts": [record["row_count"] for record in members],
                                    "accuracy_figure": bool(generated)})
    with (args.output / "performance.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("source", "row_count", "seed", "row_start", "probe_mode", "probe_stride", "flush_scope",
                         "selector", "processing", "tp", "fp", "fn", "precision", "recall", "f1", "exact_match"))
        for record in records:
            meta = record["metadata"]
            for processing in ("raw", "leading_group"):
                score = record["scores"]["methods"][args.selector][processing]
                writer.writerow((record["source"], record["row_count"], meta["seed"], meta["row_start"],
                                 meta["probe_mode"], meta["probe_stride"], meta["flush_scope"], args.selector,
                                 processing, *[score[key] for key in
                                               ("tp", "fp", "fn", "precision", "recall", "f1", "exact_match")]))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Prefill experiment figures", "",
             f"Selector: `{args.selector}`. Included measurements: {len(records)}; exclusions: {len(excluded)}.", "",
             "- `overview-*.pdf/svg/png`: Outlier Column groups, raw candidates, and first-group guesses for each acquisition.",
             "- `runs/*/groups.*`: the same comparison for one acquisition. Black ticks mark exact injected feature columns.",
             "- `runs/*/rows.*`: measured reload hit fractions by token row and 32-column group; orange outlines mark injected groups.",
             "- `accuracy-cohort-*`: precision, recall, and F1 versus observed rows, connecting only matching configurations.",
             "- `plotted_groups.csv`, `performance.csv`, `runs/*/hit_fractions.csv`, `manifest.json`: plot values, settings, source paths, and scores.", "",
             "## Figure interpretation", "",
             "Detected positions are 32-column groups, not recovered exact feature indices. The leading-group heuristic assumes "
             "forward prefetch and can discard real adjacent groups. Raw candidates remain visible. "
             "Heatmaps show repeated cache observations, not independent model inputs. Precision/recall are group-level metrics.", "",
             "Performance cohorts keep full dimensions, seed, row start, probe mode/stride, flush scope, activation threshold, "
             "outlier count, repeat count, and analysis thresholds fixed. Unrecorded settings form separate cohorts. "
             "A cohort with only one observed-row count has no accuracy curve. Means at duplicated counts use acquisitions "
             "as the averaging unit; any error bars are min/max ranges, not confidence intervals. Undefined precision/recall "
             "is plotted as zero. No uncertainty interval is invented for a single acquisition.", "",
             "## Included acquisitions", "",
             "| Source | Observed rows | Seed | Probe | Stride | Flush |", "|---|---|---|---|---|---|"]
    for record in records:
        meta = record["metadata"]
        lines.append(f"| {record['label']} | {record['row_count']} | {meta['seed']} | {meta['probe_mode']} | "
                     f"{meta['probe_stride']} | {meta['flush_scope']} |")
    lines += ["", "## Accuracy cohorts", ""]
    for cohort in manifest["cohorts"]:
        lines.append(f"- Cohort {cohort['id']}: row counts {cohort['observed_row_counts']}; "
                     f"{'curve exported' if cohort['accuracy_figure'] else 'single count; no curve'}.")
    if excluded:
        lines += ["", "## Excluded inputs", "", *[f"- {item['source']}: {item['error']}" for item in excluded]]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Generated {len(artifacts)} figure files from {len(records)} acquisitions in {args.output}")
    print(f"Figure guide: {args.output / 'README.md'}")
    return 1 if excluded else 0


if __name__ == "__main__":
    sys.exit(main())
