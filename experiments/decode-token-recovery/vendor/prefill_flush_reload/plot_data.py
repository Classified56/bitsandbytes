"""Read experiment artifacts without changing measurements or analysis rules.

All selection and validation lives here so it can be checked independently of
Matplotlib. Different seeds, scan protocols, or thresholds form separate
cohorts; their points must not silently become one observed-row-count curve.
"""

from collections import defaultdict
import json
from pathlib import Path

from .analyze import analyze_csv
from .scoring import detection_scores


COHORT_FIELDS = ("tokens", "intermediate", "hidden", "seed", "row_start", "probe_mode",
                 "probe_stride", "flush_scope", "threshold", "outliers", "repeats",
                 "row_hit_fraction", "min_row_fraction", "min_control_delta")


def discover_results(inputs, excluded=None):
    """Resolve overlapping input trees to unique measurement directories."""
    found = set()
    excluded = Path(excluded).resolve() if excluded else None
    for entry in inputs:
        entry = Path(entry).resolve()
        if not entry.exists():
            raise ValueError(f"Input does not exist: {entry}")
        if entry.is_file():
            if entry.name != "summary.json":
                raise ValueError(f"Expected a result folder or summary.json: {entry}")
            paths = [entry]
        else:
            paths = entry.rglob("summary.json")
        for path in paths:
            folder = path.parent.resolve()
            if excluded is not None and (folder == excluded or excluded in folder.parents):
                continue
            found.add(folder)
    return sorted(found)


def load_result(folder):
    """Use saved rules when present, and verify injected groups agree with truth."""
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["configuration"]
    analysis_path = folder / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        analysis_source = "saved analysis.json"
    else:
        rules = summary.get("detection_summary", {}).get("rules", {})
        analysis = analyze_csv(folder / "lines.csv", **rules)
        analysis_source = "derived from lines.csv (saved rules or analyzer defaults)"
    groups = sorted(analysis["groups"], key=lambda item: item["column_start"])
    width = int(config["intermediate"])
    if width % 32 or [group["column_start"] for group in groups] != list(range(0, width, 32)):
        raise ValueError("Expected complete, aligned 32-column groups in the report")
    rows = list(analysis["observed_rows"])
    if not rows or rows != sorted(set(rows)):
        raise ValueError("Observed rows must be nonempty, ordered, and unique")
    injected_columns = summary["cases"]["injected_gather"]["outlier_columns"]
    expected = {column // 32 * 32 for column in injected_columns}
    reported = {group["column_start"] for group in groups if group["contains_outlier"]}
    if expected != reported:
        raise ValueError("Injected columns in summary.json disagree with analysis ground truth")
    if any(not 0 <= column < width for column in injected_columns):
        raise ValueError("Injected column lies outside the activation matrix")
    if any(not 0 <= row < config["tokens"] for row in rows):
        raise ValueError("Observed row lies outside the full prefill")
    if config.get("probe_rows") is not None and config["probe_rows"] != len(rows):
        raise ValueError("Configured row count disagrees with measured rows")
    # Re-evaluate scores from the saved candidates, retaining their original
    # detection thresholds. This also supports reports predating group scoring.
    scored = detection_scores(groups)
    metadata = {key: config.get(key, "unrecorded") for key in COHORT_FIELDS}
    metadata["row_start"] = rows[0]
    metadata.update(analysis["rules"])
    fractions = []
    for row in rows:
        values = []
        for group in groups:
            detail = next((item for item in group["rows"] if item["row"] == row), None)
            if detail is None or "injected_gather" not in detail["cases"]:
                raise ValueError("Missing per-row gather evidence")
            counts = detail["cases"]["injected_gather"]
            if counts["observations"] <= 0 or not 0 <= counts["hits"] <= counts["observations"]:
                raise ValueError("Invalid hit/observation counts")
            # Derive from integer counts rather than trusting rounded fractions.
            values.append(counts["hits"] / counts["observations"])
        fractions.append(values)
    return {"source": str(folder.resolve()), "label": folder.name, "width": width,
            "rows": rows, "row_count": len(rows), "metadata": metadata,
            "analysis_source": analysis_source, "groups": groups, "hit_fractions": fractions,
            "injected_columns": injected_columns, "scores": scored,
            "cache_check_passed": summary.get("cache_check_passed")}


def compatible_cohorts(records):
    cohorts = defaultdict(list)
    for record in records:
        key = tuple(record["metadata"].get(field, "unrecorded") for field in COHORT_FIELDS)
        cohorts[key].append(record)
    # Rows can be compared only when they are nested prefixes from the same
    # recorded start. Nonconsecutive custom row selections require their own key.
    split = defaultdict(list)
    for key, members in cohorts.items():
        for record in members:
            standard = record["rows"] == list(range(record["rows"][0], record["rows"][0] + record["row_count"]))
            split[(key, None if standard else tuple(record["rows"]))].append(record)
    return [sorted(members, key=lambda item: (item["row_count"], item["source"]))
            for _, members in sorted(split.items(), key=lambda item: repr(item[0]))]
