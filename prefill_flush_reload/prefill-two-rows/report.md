# Cache-line detection report

Observed token rows (2): 30-31.

Ranges below are 32-column cache-line groups, with an exclusive upper bound. They do not identify an exact feature column.

A row is consistent when >= 60% of its repeats hit. A group is detected when >= 50% of observed rows are consistent. Control-adjusted rows additionally exceed both controls by >= 50%.

## Detected groups and row identities

The table includes consensus detections and injected groups, including any missed groups. Exact per-row hit counts, all one-off hits, and all undetected groups appear in `row_sections.csv` and `analysis.json`.

| Columns | Injected group | Any-hit rows (count; IDs) | Consistent rows (count; IDs) | Adjusted rows (count; IDs) | Detected / adjusted |
|---|---|---|---|---|---|
| 192:224 | yes | 2; 30-31 | 2; 30-31 | 2; 30-31 | True / True |
| 896:928 | yes | 2; 30-31 | 2; 30-31 | 2; 30-31 | True / True |
| 928:960 | no | 2; 30-31 | 1; 30 | 1; 30 | True / True |
| 1984:2016 | yes | 2; 30-31 | 2; 30-31 | 2; 30-31 | True / True |
| 2240:2272 | yes | 2; 30-31 | 2; 30-31 | 2; 30-31 | True / True |
| 2272:2304 | no | 2; 30-31 | 2; 30-31 | 2; 30-31 | True / True |
| 2304:2336 | no | 2; 30-31 | 1; 30 | 1; 30 | True / True |

## Spatial halo diagnostic

Hit rates around known injected groups; other true outlier groups are excluded at nonzero offsets. Positive offsets mean higher column indices in the SAME row, not later probes. These labels evaluate the footprint and do not filter detections.

| Offset (groups) | No outliers | Gather | Skipped gather |
|---|---|---|---|
| -4 | 0.00% | 0.00% | 0.00% |
| -3 | 0.00% | 0.00% | 0.00% |
| -2 | 0.00% | 0.00% | 0.00% |
| -1 | 0.00% | 0.00% | 0.00% |
| 0 | 0.00% | 100.00% | 0.00% |
| 1 | 0.00% | 32.50% | 0.00% |
| 2 | 0.00% | 27.50% | 0.00% |
| 3 | 0.00% | 0.00% | 0.00% |
| 4 | 0.00% | 0.00% | 0.00% |
| 5 | 0.00% | 0.00% | 0.00% |
| 6 | 0.00% | 0.00% | 0.00% |
| 7 | 0.00% | 0.00% | 0.00% |
| 8 | 0.00% | 0.00% | 0.00% |

## Probe-order diagnostic

Non-accessed lines only. Nearby means within four groups in the same row. This association is not proof of causation; old CSVs have no probe-order metadata.

| Case | Previous probe condition | False-hit rate | Observations |
|---|---|---|---|
| injected_gather | first_probe_in_window | 1.62% | 1480 |
| injected_no_gather | first_probe_in_window | 0.00% | 1520 |
| no_outliers | first_probe_in_window | 0.00% | 1520 |

## Interpretation

Consensus can reject occasional noise. Control adjustment can reject background hits shared with the no-gather controls. Neither removes a repeatable halo caused by the gather itself. Adjacent detections remain ambiguous; they are also grouped as `ambiguous_contiguous_clusters` in JSON rather than silently deleted.

A row subset parsed from an old CSV still contains interference from the original full scan. Use an actual row-limited run to change the measurement protocol.
