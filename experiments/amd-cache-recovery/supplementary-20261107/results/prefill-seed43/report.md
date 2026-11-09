# Cache-line detection report

Observed token rows (64): 0-63.

Ranges below are 32-column cache-line groups, with an exclusive upper bound. They do not identify an exact feature column.

A row is consistent when >= 60% of its repeats hit. A group is detected when >= 50% of observed rows are consistent. Control-adjusted rows additionally exceed both controls by >= 50%.

## Guessed groups: first candidate in each consecutive run

The primary guesses below keep only the lowest-column group of each consecutive candidate run. Following candidates remain listed as possible prefetch hits. A run is formed by exact 32-column adjacency, without bridging gaps. This heuristic can suppress genuinely adjacent outlier groups.

Scores compare 32-column groups, not exact individual feature columns. Multiple real columns in one group count as one actual group. Precision/recall/F1 and exact-set match are more useful than overall accuracy when true groups are sparse.

| Selector | Processing | TP | FP | FN | Precision | Recall | F1 | Accuracy | Exact set |
|---|---|---|---|---|---|---|---|---|---|
| consensus | raw | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |
| consensus | leading_group | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |
| control_adjusted | raw | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |
| control_adjusted | leading_group | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |

| Selector | Guessed columns | Following candidate groups (starts) | Actual guess? | Real groups suppressed |
|---|---|---|---|---|
| consensus | 288:320 | [] | True | [] |
| consensus | 1152:1184 | [] | True | [] |
| consensus | 2336:2368 | [] | True | [] |
| consensus | 3776:3808 | [] | True | [] |
| control_adjusted | 288:320 | [] | True | [] |
| control_adjusted | 1152:1184 | [] | True | [] |
| control_adjusted | 2336:2368 | [] | True | [] |
| control_adjusted | 3776:3808 | [] | True | [] |

## Raw candidate groups and row identities

The table includes consensus detections and injected groups, including any missed groups. Exact per-row hit counts, all one-off hits, and all undetected groups appear in `row_sections.csv` and `analysis.json`.

| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |
|---|---|---|---|---|---|
| 288:320 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 1152:1184 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 2336:2368 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 3776:3808 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 0.08% | 0.00% | 0.00% |
| -3 | 0.08% | 0.00% | 0.00% |
| -2 | 0.00% | 0.31% | 0.00% |
| -1 | 0.00% | 0.00% | 0.00% |
| 0 | 0.16% | 99.77% | 0.00% |
| 1 | 0.16% | 5.78% | 0.00% |
| 2 | 0.00% | 6.17% | 0.00% |
| 3 | 0.00% | 0.00% | 0.00% |
| 4 | 0.16% | 0.00% | 0.00% |
| 5 | 0.16% | 0.00% | 0.00% |
| 6 | 0.00% | 0.00% | 0.00% |
| 7 | 0.00% | 0.00% | 0.00% |
| 8 | 0.08% | 0.00% | 0.00% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 0.29% | 1383 |
| injected_gather | after_miss | 0.34% | 45955 |
| injected_gather | after_nearby_hit | 0.00% | 3 |
| injected_gather | first_probe_in_window | 0.00% | 19 |
| injected_no_gather | after_distant_hit | 0.00% | 1 |
| injected_no_gather | after_miss | 0.00% | 48619 |
| injected_no_gather | first_probe_in_window | 0.00% | 20 |
| no_outliers | after_distant_hit | 0.00% | 25 |
| no_outliers | after_miss | 0.05% | 48595 |
| no_outliers | first_probe_in_window | 0.00% | 20 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
