"""Group consecutive candidates and score guesses at cache-line resolution.

The leading-group heuristic uses candidate positions only. Ground truth is
consulted afterwards to score it, including true neighbors it may suppress.
"""


def candidate_runs(candidates):
    """Keep the lowest-column group as the guess; retain every following hit."""
    runs = []
    for start in sorted(set(candidates)):
        if runs and start == runs[-1]["candidate_groups"][-1] + 32:
            runs[-1]["candidate_groups"].append(start)
            runs[-1]["following_groups"].append(start)
            runs[-1]["column_end_exclusive"] = start + 32
        else:
            runs.append({"guessed_group": start, "column_start": start,
                         "column_end_exclusive": start + 32,
                         "candidate_groups": [start], "following_groups": []})
    return runs


def score_groups(predicted, actual, universe):
    """Score sets of 32-column group starts, avoiding sparse accuracy inflation."""
    predicted, actual, universe = set(predicted), set(actual), set(universe)
    if not predicted <= universe or not actual <= universe:
        raise ValueError("Predicted and actual groups must be in the observed universe")
    matched, extra, missed = predicted & actual, predicted - actual, actual - predicted
    tp, fp, fn = len(matched), len(extra), len(missed)
    tn = len(universe - (predicted | actual))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"predicted_groups": sorted(predicted), "actual_groups": sorted(actual),
            "matched_groups": sorted(matched), "false_positive_groups": sorted(extra),
            "missed_groups": sorted(missed), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "jaccard": tp / len(predicted | actual) if predicted | actual else None,
            "accuracy": (tp + tn) / len(universe) if universe else None,
            "exact_match": predicted == actual}


def detection_scores(groups):
    """Evaluate raw and first-in-run guesses independently for both selectors."""
    universe = [group["column_start"] for group in groups]
    actual = [group["column_start"] for group in groups if group["contains_outlier"]]
    results = {"resolution_columns": 32, "actual_groups": actual,
               "heuristic": "first group in each strictly consecutive candidate run",
               "methods": {}}
    for name, flag in (("consensus", "detected"), ("control_adjusted", "control_adjusted_detected")):
        candidates = [group["column_start"] for group in groups if group[flag]]
        runs = candidate_runs(candidates)
        guesses = [run["guessed_group"] for run in runs]
        # Annotate evaluation after constructing runs. Never change guesses
        # using truth: real adjacent features must be allowed to become misses.
        for run in runs:
            run["guessed_group_is_actual"] = run["guessed_group"] in actual
            run["following_actual_groups"] = [start for start in run["following_groups"] if start in actual]
        results["methods"][name] = {"raw": score_groups(candidates, actual, universe),
                                    "leading_group": score_groups(guesses, actual, universe),
                                    "candidate_runs": runs}
    return results
