# Cache-line detection report

Observed token rows (32): 0-31.

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
| 192:224 | yes | 32; 0-31 | 32; 0-31 | 32; 0-31 | True / True |
| 896:928 | yes | 32; 0-31 | 32; 0-31 | 32; 0-31 | True / True |
| 928:960 | no | 30; 0-10,12-30 | 17; 0-1,3-5,7-9,12,14,16-17,19,22,26-27,30 | 17; 0-1,3-5,7-9,12,14,16-17,19,22,26-27,30 | True / True |
| 960:992 | no | 31; 0-17,19-31 | 22; 0-3,5,9,12-14,16-17,19-20,22,24-31 | 22; 0-3,5,9,12-14,16-17,19-20,22,24-31 | True / True |
| 1984:2016 | yes | 32; 0-31 | 32; 0-31 | 32; 0-31 | True / True |
| 2240:2272 | yes | 32; 0-31 | 32; 0-31 | 32; 0-31 | True / True |
| 2272:2304 | no | 30; 0-7,9-23,25-31 | 23; 0-2,4-6,10-11,13-15,17-23,26-29,31 | 23; 0-2,4-6,10-11,13-15,17-23,26-29,31 | True / True |
| 2304:2336 | no | 25; 1-7,9-12,14-15,17-20,22-23,25-28,30-31 | 21; 1-2,4,6-7,9-12,14-15,17-20,23,25-27,30-31 | 21; 1-2,4,6-7,9-12,14-15,17-20,23,25-27,30-31 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 0.16% | 2.19% | 0.16% |
| -3 | 0.00% | 1.09% | 0.00% |
| -2 | 0.31% | 1.88% | 0.31% |
| -1 | 0.16% | 3.28% | 0.16% |
| 0 | 0.31% | 99.69% | 0.00% |
| 1 | 0.00% | 39.22% | 0.00% |
| 2 | 0.16% | 36.56% | 0.16% |
| 3 | 0.16% | 0.47% | 0.62% |
| 4 | 0.47% | 0.00% | 0.16% |
| 5 | 0.16% | 0.31% | 0.00% |
| 6 | 0.16% | 0.47% | 0.16% |
| 7 | 0.16% | 0.62% | 0.16% |
| 8 | 0.21% | 0.21% | 0.21% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | after_distant_hit | 2.74% | 1205 |
| injected_gather | after_miss | 2.50% | 22450 |
| injected_gather | first_probe_in_window | 4.00% | 25 |
| injected_no_gather | after_distant_hit | 0.00% | 39 |
| injected_no_gather | after_miss | 0.16% | 24256 |
| injected_no_gather | first_probe_in_window | 0.00% | 25 |
| no_outliers | after_distant_hit | 0.00% | 54 |
| no_outliers | after_miss | 0.22% | 24241 |
| no_outliers | first_probe_in_window | 0.00% | 25 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. The leading-group guesses apply a forward-halo assumption; raw detections, following candidates, and any real groups suppressed by that assumption remain visible. Ground truth evaluates guesses but never selects them.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
