# Prefill experiment figures

Selector: `consensus`. Included measurements: 9; exclusions: 0.

- `overview-*.pdf/svg/png`: Outlier Column groups, raw candidates, and first-group guesses for each acquisition.
- `runs/*/groups.*`: the same comparison for one acquisition. Black ticks mark exact injected feature columns.
- `runs/*/rows.*`: measured reload hit fractions by token row and 32-column group; orange outlines mark injected groups.
- `accuracy-cohort-*`: precision, recall, and F1 versus observed rows, connecting only matching configurations.
- `plotted_groups.csv`, `performance.csv`, `runs/*/hit_fractions.csv`, `manifest.json`: plot values, settings, source paths, and scores.

## Figure interpretation

Detected positions are 32-column groups, not recovered exact feature indices. The leading-group heuristic assumes forward prefetch and can discard real adjacent groups. Raw candidates remain visible. Heatmaps show repeated cache observations, not independent model inputs. Precision/recall are group-level metrics.

Performance cohorts keep full dimensions, seed, row start, probe mode/stride, flush scope, activation threshold, outlier count, repeat count, and analysis thresholds fixed. Unrecorded settings form separate cohorts. A cohort with only one observed-row count has no accuracy curve. Means at duplicated counts use acquisitions as the averaging unit; any error bars are min/max ranges, not confidence intervals. Undefined precision/recall is plotted as zero. No uncertainty interval is invented for a single acquisition.

## Included acquisitions

| Source | Observed rows | Seed | Probe | Stride | Flush |
|---|---|---|---|---|---|
| prefill-16-rows | 16 | 42 | sweep | 5 | observed |
| prefill-2-rows | 2 | 420 | sweep | 5 | observed |
| prefill-32-rows | 32 | 42 | sweep | 5 | observed |
| prefill-4-rows | 4 | 420 | sweep | 5 | observed |
| prefill-8-rows | 8 | 42 | sweep | 5 | observed |
| prefill-all-rows | 64 | 42 | sweep | 5 | all |
| prefill-four-rows | 4 | 42 | sweep | 5 | observed |
| prefill-results | 64 | 42 | unrecorded | 7 | unrecorded |
| prefill-two-rows | 2 | 42 | single | 4 | all |

## Accuracy cohorts

- Cohort 1: row counts [64]; single count; no curve.
- Cohort 2: row counts [4, 8, 16, 32]; curve exported.
- Cohort 3: row counts [64]; single count; no curve.
- Cohort 4: row counts [2]; single count; no curve.
- Cohort 5: row counts [2, 4]; curve exported.
