# Cache-line detection report

Observed token rows (4): 30-33.

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
| consensus | 192:224 | [] | True | [] |
| consensus | 896:928 | [] | True | [] |
| consensus | 1984:2016 | [] | True | [] |
| consensus | 2240:2272 | [] | True | [] |
| control_adjusted | 192:224 | [] | True | [] |
| control_adjusted | 896:928 | [] | True | [] |
| control_adjusted | 1984:2016 | [] | True | [] |
| control_adjusted | 2240:2272 | [] | True | [] |

## Raw candidate groups and row identities

The table includes consensus detections and injected groups, including any missed groups. Exact per-row hit counts, all one-off hits, and all undetected groups appear in `row_sections.csv` and `analysis.json`.

| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |
|---|---|---|---|---|---|
| 192:224 | yes | 4; 30-33 | 4; 30-33 | 4; 30-33 | True / True |
| 896:928 | yes | 4; 30-33 | 4; 30-33 | 4; 30-33 | True / True |
| 1984:2016 | yes | 4; 30-33 | 4; 30-33 | 4; 30-33 | True / True |
| 2240:2272 | yes | 4; 30-33 | 4; 30-33 | 4; 30-33 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 3.75% | 2.50% | 2.50% |
| -3 | 7.50% | 8.75% | 11.25% |
| -2 | 3.75% | 7.50% | 5.00% |
| -1 | 3.75% | 5.00% | 6.25% |
| 0 | 5.00% | 100.00% | 3.75% |
| 1 | 8.75% | 6.25% | 5.00% |
| 2 | 6.25% | 3.75% | 3.75% |
| 3 | 2.50% | 6.25% | 2.50% |
| 4 | 3.75% | 5.00% | 3.75% |
| 5 | 7.50% | 10.00% | 10.00% |
| 6 | 7.50% | 7.50% | 5.00% |
| 7 | 2.50% | 6.25% | 5.00% |
| 8 | 10.00% | 5.00% | 6.67% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 10.71% | 252 |
| injected_gather | after_miss | 5.73% | 2687 |
| injected_gather | after_nearby_hit | 0.00% | 2 |
| injected_gather | first_probe_in_window | 0.00% | 19 |
| injected_no_gather | after_distant_hit | 18.23% | 181 |
| injected_no_gather | after_miss | 5.36% | 2838 |
| injected_no_gather | after_nearby_hit | 0.00% | 1 |
| injected_no_gather | first_probe_in_window | 0.00% | 20 |
| no_outliers | after_distant_hit | 19.88% | 166 |
| no_outliers | after_miss | 4.80% | 2854 |
| no_outliers | first_probe_in_window | 0.00% | 20 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
