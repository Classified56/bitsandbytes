# Cache-line detection report

Observed token rows (2): 0-1.

Ranges below are 32-column cache-line groups, with an exclusive upper bound. They do not identify an exact feature column.

A row is consistent when >= 60% of its repeats hit. A group is detected when >= 50% of observed rows are consistent. Control-adjusted rows additionally exceed both controls by >= 50%.

## Guessed groups: first candidate in each consecutive run

The primary guesses below keep only the lowest-column group of each consecutive candidate run. Following candidates remain listed as possible prefetch hits. A run is formed by exact 32-column adjacency, without bridging gaps. This heuristic can suppress genuinely adjacent outlier groups.

Scores compare 32-column groups, not exact individual feature columns. Multiple real columns in one group count as one actual group. Precision/recall/F1 and exact-set match are more useful than overall accuracy when true groups are sparse.

| Selector | Processing | TP | FP | FN | Precision | Recall | F1 | Accuracy | Exact set |
|---|---|---|---|---|---|---|---|---|---|
| consensus | raw | 4 | 10 | 0 | 28.57% | 100.00% | 44.44% | 93.42% | False |
| consensus | leading_group | 1 | 4 | 3 | 20.00% | 25.00% | 22.22% | 95.39% | False |
| control_adjusted | raw | 4 | 10 | 0 | 28.57% | 100.00% | 44.44% | 93.42% | False |
| control_adjusted | leading_group | 1 | 4 | 3 | 20.00% | 25.00% | 22.22% | 95.39% | False |

| Selector | Guessed columns | Following candidate groups (starts) | Actual guess? | Real groups suppressed |
|---|---|---|---|---|
| consensus | 192:224 | [224] | True | [] |
| consensus | 1280:1312 | [1312] | False | [] |
| consensus | 2176:2208 | [2208] | False | [2208] |
| consensus | 2944:2976 | [2976, 3008, 3040] | False | [2976] |
| consensus | 3264:3296 | [3296, 3328, 3360] | False | [3296] |
| control_adjusted | 192:224 | [224] | True | [] |
| control_adjusted | 1280:1312 | [1312] | False | [] |
| control_adjusted | 2176:2208 | [2208] | False | [2208] |
| control_adjusted | 2944:2976 | [2976, 3008, 3040] | False | [2976] |
| control_adjusted | 3264:3296 | [3296, 3328, 3360] | False | [3296] |

## Raw candidate groups and row identities

The table includes consensus detections and injected groups, including any missed groups. Exact per-row hit counts, all one-off hits, and all undetected groups appear in `row_sections.csv` and `analysis.json`.

| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |
|---|---|---|---|---|---|
| 192:224 | yes | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 224:256 | no | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 1280:1312 | no | 2; 0-1 | 1; 1 | 1; 1 | True / True |
| 1312:1344 | no | 1; 1 | 1; 1 | 1; 1 | True / True |
| 2176:2208 | no | 2; 0-1 | 1; 0 | 1; 0 | True / True |
| 2208:2240 | yes | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 2944:2976 | no | 2; 0-1 | 1; 0 | 1; 0 | True / True |
| 2976:3008 | yes | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 3008:3040 | no | 2; 0-1 | 2; 0-1 | 1; 0 | True / True |
| 3040:3072 | no | 2; 0-1 | 1; 1 | 1; 1 | True / True |
| 3264:3296 | no | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 3296:3328 | yes | 2; 0-1 | 2; 0-1 | 2; 0-1 | True / True |
| 3328:3360 | no | 1; 0 | 1; 0 | 1; 0 | True / True |
| 3360:3392 | no | 1; 0 | 1; 0 | 1; 0 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 5.00% | 7.50% | 5.00% |
| -3 | 2.50% | 5.00% | 2.50% |
| -2 | 0.00% | 2.50% | 0.00% |
| -1 | 0.00% | 52.50% | 0.00% |
| 0 | 0.00% | 100.00% | 0.00% |
| 1 | 5.00% | 65.00% | 5.00% |
| 2 | 5.00% | 30.00% | 5.00% |
| 3 | 10.00% | 7.50% | 10.00% |
| 4 | 2.50% | 2.50% | 2.50% |
| 5 | 0.00% | 2.50% | 0.00% |
| 6 | 2.50% | 5.00% | 2.50% |
| 7 | 5.00% | 5.00% | 5.00% |
| 8 | 0.00% | 0.00% | 0.00% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 7.41% | 135 |
| injected_gather | after_miss | 7.42% | 1321 |
| injected_gather | first_probe_in_window | 0.00% | 24 |
| injected_no_gather | after_distant_hit | 0.00% | 42 |
| injected_no_gather | after_miss | 3.23% | 1453 |
| injected_no_gather | first_probe_in_window | 0.00% | 25 |
| no_outliers | after_distant_hit | 0.00% | 43 |
| no_outliers | after_miss | 3.31% | 1452 |
| no_outliers | first_probe_in_window | 0.00% | 25 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
