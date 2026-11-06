# Cache-line detection report

Observed token rows (8): 0-7.

Ranges below are 32-column cache-line groups, with an exclusive upper bound. They do not identify an exact feature column.

A row is consistent when >= 60% of its repeats hit. A group is detected when >= 50% of observed rows are consistent. Control-adjusted rows additionally exceed both controls by >= 50%.

## Guessed groups: first candidate in each consecutive run

The primary guesses below keep only the lowest-column group of each consecutive candidate run. Following candidates remain listed as possible prefetch hits. A run is formed by exact 32-column adjacency, without bridging gaps. This heuristic can suppress genuinely adjacent outlier groups.

Scores compare 32-column groups, not exact individual feature columns. Multiple real columns in one group count as one actual group. Precision/recall/F1 and exact-set match are more useful than overall accuracy when true groups are sparse.

| Selector | Processing | TP | FP | FN | Precision | Recall | F1 | Accuracy | Exact set |
|---|---|---|---|---|---|---|---|---|---|
| consensus | raw | 4 | 4 | 0 | 50.00% | 100.00% | 66.67% | 97.37% | False |
| consensus | leading_group | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |
| control_adjusted | raw | 4 | 4 | 0 | 50.00% | 100.00% | 66.67% | 97.37% | False |
| control_adjusted | leading_group | 4 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | True |

| Selector | Guessed columns | Following candidate groups (starts) | Actual guess? | Real groups suppressed |
|---|---|---|---|---|
| consensus | 192:224 | [] | True | [] |
| consensus | 896:928 | [928, 960] | True | [] |
| consensus | 1984:2016 | [] | True | [] |
| consensus | 2240:2272 | [2272, 2304] | True | [] |
| control_adjusted | 192:224 | [] | True | [] |
| control_adjusted | 896:928 | [928, 960] | True | [] |
| control_adjusted | 1984:2016 | [] | True | [] |
| control_adjusted | 2240:2272 | [2272, 2304] | True | [] |

## Raw candidate groups and row identities

The table includes consensus detections and injected groups, including any missed groups. Exact per-row hit counts, all one-off hits, and all undetected groups appear in `row_sections.csv` and `analysis.json`.

| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |
|---|---|---|---|---|---|
| 192:224 | yes | 8; 0-7 | 8; 0-7 | 8; 0-7 | True / True |
| 896:928 | yes | 8; 0-7 | 8; 0-7 | 8; 0-7 | True / True |
| 928:960 | no | 7; 0-4,6-7 | 6; 0-4,6 | 5; 0-4 | True / True |
| 960:992 | no | 7; 0,2-7 | 6; 0,2-6 | 6; 0,2-6 | True / True |
| 1984:2016 | yes | 8; 0-7 | 8; 0-7 | 8; 0-7 | True / True |
| 2240:2272 | yes | 8; 0-7 | 8; 0-7 | 8; 0-7 | True / True |
| 2272:2304 | no | 8; 0-7 | 7; 0-6 | 7; 0-6 | True / True |
| 2304:2336 | no | 6; 1-4,6-7 | 4; 1-2,6-7 | 4; 1-2,6-7 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 1.88% | 3.12% | 1.88% |
| -3 | 3.75% | 5.62% | 3.75% |
| -2 | 1.88% | 4.38% | 2.50% |
| -1 | 1.88% | 3.75% | 2.50% |
| 0 | 4.38% | 100.00% | 4.38% |
| 1 | 3.75% | 47.50% | 3.12% |
| 2 | 2.50% | 40.62% | 1.88% |
| 3 | 5.00% | 3.12% | 5.62% |
| 4 | 4.38% | 3.12% | 5.00% |
| 5 | 3.12% | 1.88% | 2.50% |
| 6 | 1.25% | 2.50% | 1.88% |
| 7 | 4.38% | 4.38% | 3.75% |
| 8 | 4.17% | 3.33% | 3.33% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 7.78% | 463 |
| injected_gather | after_miss | 5.12% | 5432 |
| injected_gather | first_probe_in_window | 8.00% | 25 |
| injected_no_gather | after_distant_hit | 5.41% | 185 |
| injected_no_gather | after_miss | 2.98% | 5870 |
| injected_no_gather | first_probe_in_window | 0.00% | 25 |
| no_outliers | after_distant_hit | 3.14% | 191 |
| no_outliers | after_miss | 3.15% | 5864 |
| no_outliers | first_probe_in_window | 0.00% | 25 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
