# Cache-line detection report

Observed token rows (64): 0-63.

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
| 192:224 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 896:928 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 928:960 | no | 61; 0-14,16-25,27-28,30-63 | 35; 0-1,3-8,11-12,14,17,23-24,31-33,36,40-42,46,48-54,56-57,59-62 | 35; 0-1,3-8,11-12,14,17,23-24,31-33,36,40-42,46,48-54,56-57,59-62 | True / True |
| 960:992 | no | 64; 0-63 | 42; 0-4,6,8,11-12,14-20,22,24,28-30,32-34,36,38,40-41,44-46,48-51,54-57,60,62-63 | 42; 0-4,6,8,11-12,14-20,22,24,28-30,32-34,36,38,40-41,44-46,48-51,54-57,60,62-63 | True / True |
| 1984:2016 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 2240:2272 | yes | 64; 0-63 | 64; 0-63 | 64; 0-63 | True / True |
| 2272:2304 | no | 54; 0-7,9-15,17-20,22-23,25-28,30-31,33-39,41-47,49-55,57-60,62-63 | 45; 1-3,5-7,10-11,14-15,17-20,23,25-28,30-31,34-39,41-42,45-47,49-55,57-60,62-63 | 45; 1-3,5-7,10-11,14-15,17-20,23,25-28,30-31,34-39,41-42,45-47,49-55,57-60,62-63 | True / True |
| 2304:2336 | no | 48; 1-4,6-7,9-12,14-15,17-20,22-23,25-28,30-31,33-36,38-39,41-44,46-47,49-52,54-55,57-60,62-63 | 37; 2-3,6-7,9-12,14-15,17-20,22-23,25-26,28,31,33-34,36,38-39,41-43,49-52,55,57-58,60,63 | 37; 2-3,6-7,9-12,14-15,17-20,22-23,25-26,28,31,33-34,36,38-39,41-43,49-52,55,57-58,60,63 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 0.08% | 0.70% | 0.00% |
| -3 | 0.00% | 0.70% | 0.00% |
| -2 | 0.00% | 0.94% | 0.08% |
| -1 | 0.00% | 0.94% | 0.00% |
| 0 | 0.08% | 98.59% | 0.31% |
| 1 | 0.00% | 38.12% | 0.00% |
| 2 | 0.08% | 34.45% | 0.00% |
| 3 | 0.08% | 0.08% | 0.00% |
| 4 | 0.16% | 0.23% | 0.16% |
| 5 | 0.00% | 0.08% | 0.00% |
| 6 | 0.00% | 0.08% | 0.00% |
| 7 | 0.00% | 0.16% | 0.00% |
| 8 | 0.00% | 0.00% | 0.00% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 2.24% | 2229 |
| injected_gather | after_miss | 2.18% | 45106 |
| injected_gather | first_probe_in_window | 4.00% | 25 |
| injected_no_gather | after_distant_hit | 0.00% | 23 |
| injected_no_gather | after_miss | 0.05% | 48592 |
| injected_no_gather | first_probe_in_window | 0.00% | 25 |
| no_outliers | after_distant_hit | 0.00% | 26 |
| no_outliers | after_miss | 0.05% | 48589 |
| no_outliers | first_probe_in_window | 0.00% | 25 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
