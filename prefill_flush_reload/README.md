# Prefill FFN Flush+Reload simulation

This is a controlled, single-process experiment with synthetic activations,
not a pretrained model or an observation of another application. It measures
actual cache reload times around a sparse FP16 activation gather.

Run from the repository root:

```bash
python -m prefill_flush_reload --output /tmp/prefill-results
```

No Python packages are required. The helper compiles into a temporary directory
using `cc` (override with `CC`). It requires x86-64, F16C, RDTSCP, and 64-byte
cache lines. The CPU capabilities are checked before executing the kernels.
The existing `flush_reload_native.c` supplies the fenced timer. Measurements
are **TSC ticks**, not nanoseconds or CPU core clock cycles.

The default is a **64-token prefill**, with an activation matrix `[64, 4864]`
and a down-projection weight matrix `[896, 4864]`. These feature dimensions
come from [Qwen2.5-0.5B-Instruct's configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json).
The experiment deliberately uses FP16 activation storage even though that
checkpoint declares BF16. It does not reproduce a trained activation
distribution, the preceding SwiGLU, or model accuracy.

Customize the prefill and layer dimensions:

```bash
python -m prefill_flush_reload --tokens 128 --intermediate 4864 --hidden 896 \
    --outliers 8 --repeats 10 --output /tmp/prefill-128
```

Tokens must be at least two, and the intermediate dimension must be a multiple
of 32. Larger projections increase reference-computation time and memory use.
Small toy shapes can have substantially more prefetch interference and are
useful for numerical debugging rather than representative cache measurements.

## Execution and observation window

1. Generate bounded FP16 synthetic post-FFN activations and FP16 weights.
2. Optionally inject threshold-crossing values in a few reproducible columns,
   across token rows. Detect the union of columns with `abs(A) >= threshold`.
3. Quantize activations with per-row scales and suppress whole selected columns
   in INT8. Quantize weights to INT8 with per-output-feature scales.
4. Flush every line of the original FP16 activation matrix.
5. Gather the selected FP16 columns into a separate, prefaulted `subA` buffer.
6. Reload monitored source lines before any output computation or dequantization.
7. After all cache observations, compute normal and mixed outputs, reconstructing
   outlier weights from INT8 plus scales and rounding them to FP16. Independently
   check sampled outputs in Python and compare full outputs against two references.

Calls run sequentially: the return from each native call is the synchronization
boundary. Independent sender/receiver threads and slot windows are unnecessary
for this instrumented experiment. Quantization and ground-truth inspection
occur outside the measured window.

For a multi-token prefill, **every row's entry in a selected column is gathered**,
even if just one row crossed the activation threshold. A 64-byte FP16 cache line
covers 32 consecutive columns within a row. The row stride is cache-line aligned.

The normal computation here uses actual INT8 products with INT32 accumulation,
then dequantizes and rounds to FP16. Correction uses FP32 accumulation and FP16
output rounding. This matches the decomposition and storage concepts; it is
not a claim of bit-exact PyTorch arithmetic. In particular, this host's actual
bitsandbytes CPU backend uses an FP32 fallback for nominal INT8 multiplication.

## Controls, prefetching, and measurement limits

Each run includes:

- **no_outliers**: quantization plus an empty gather; expected source accesses are zero.
- **injected_gather**: quantization plus the selected FP16 reads and numerical correction.
- **injected_no_gather**: the same injected data and quantization, with the gather
  deliberately omitted. Numerical correction is not evaluated for this control.

Timing calibration and its held-out validation use a separate page; case labels
are never used to choose the reload threshold. Both controls measure false hits.

A reload can itself prefetch another line. By default, each reconstructed trial
uses four sweeps: probe shuffled lines `0,4,8,...`, then `1,5,9,...`, etc., with
**a fresh full flush and gather for each sweep**. Every source line is measured
once per reconstructed trial, but this is not one simultaneous snapshot.
`--probe-stride 1` exposes the less controlled full scan for comparison.

Spacing probes reduces interference; it does not disable prefetchers. The gather
can itself fetch adjacent lines. Thus a cached line need not contain an outlier,
and precision can be low even with high recall when outliers are sparse. Large
matrices may also evict earlier gathered lines. These effects remain visible
in the report rather than being removed using the known outlier labels.

## Reading the results

`summary.json` contains settings, timing calibration, injected column indices,
per-case precision/recall/FPR, latency percentiles, and hit fractions for each
32-column section across all token rows and repetitions. A section's exact
outlier label is ground truth; its hit fraction is measured evidence, not an
inferred exact column identity.

`lines.csv` contains each row/section/repetition's measured latency, predicted
hit, expected source access, and whether that line contains a selected column.
The skipped-gather control has outlier lines but **no expected source accesses**.

Numerical reporting includes maximum absolute and relative L2 error against:

- FP32 computation with reconstructed FP16-rounded INT8 weights, to expose
  activation quantization and arithmetic error.
- FP32 computation with original FP16 weights, to include weight quantization error.

The normal-only output is compared to the reconstructed-weight reference too,
so the effect of adding the correction is visible. These are per-layer numerical
errors, not language-model quality metrics.

Exit status is `0` when held-out calibration accuracy and gather recall reach
90% and all case false-positive rates stay at or below 10%; `1` when timing
criteria fail; `2` for a handled setup error. Numerical assertions fail loudly.
A timing failure is an experimental result; inspect the distributions and
controls before changing any criterion. Precision is reported independently.

## File map and verification

- `__main__.py`: CLI, controls, and artifact writing.
- `simulation.py`: tensor setup, column ground truth, and numerical checks.
- `measurement.py`: calibration, staged observation, and cache metrics.
- `native.py`: compilation, ctypes signatures, and aligned buffer ownership.
- `kernels.c`: quantization, sparse gather, probes, and down-projection arithmetic.
- `test_simulation.py`: deterministic semantics and numerical tests.

```bash
python -m unittest prefill_flush_reload.test_simulation -v
```

These tests deliberately do not assert hardware timing results. Existing
`flush_reload_test.py` and its independent primitive checks remain unchanged.

## Row-limited measurement and prefetch diagnosis

Keep `--tokens 64` to simulate the full prefill, but observe just 1–4 token rows:

```bash
python -m prefill_flush_reload --tokens 64 --probe-rows 2 --row-start 30 \
    --output /tmp/prefill-two-rows
```

Token rows are zero-based; this example reloads rows 30 and 31. Quantization,
outlier-column selection, and the FP16 gather still operate on **all 64 rows**.
The default `--flush-scope all` also keeps flushing the entire activation
matrix, while only the requested rows are reloaded. Metrics exclude unobserved
rows entirely; an unobserved row is never counted as a cache miss.

To flush only the observed rows as an additional diagnostic:

```bash
python -m prefill_flush_reload --probe-rows 4 --flush-scope observed \
    --output /tmp/prefill-four-rows-local-flush
```

Other source rows remain warm in this mode, so it changes the cache preparation
and is not directly interchangeable with a full-matrix flush. Their full gather
can still influence the observed rows. Row-limiting by itself does not guarantee
lower interference: fewer candidate probes can make their sequence more
predictable, and the gather can train prefetchers independently of the reloads.

### Distinguishing the gather from the scan

Use one reload per fresh flush/gather to eliminate earlier reloads in the SAME
window as a cause of the target line being cached:

```bash
python -m prefill_flush_reload --probe-rows 2 --row-start 30 --probe-mode single \
    --repeats 5 --output /tmp/prefill-two-rows-single
```

This is slower because it executes one full flush/gather per observed line.
It is a reconstruction across repeated gathers, not a simultaneous snapshot
of one gather. It does not reset prefetcher training between windows, and the
many repeated gathers can themselves train hardware predictors.

Compare both no-gather controls between `sweep` and `single`. A background
false-positive reduction in single mode supports interference during scanning.
Neighboring hits that remain specific to the gather in single mode support
a gather-associated footprint, rather than contamination exclusively from
previous probes. Neither comparison identifies a particular hardware unit.

The [Intel optimization manual](https://www.intel.com/content/dam/doc/manual/64-ia-32-architectures-optimization-manual.pdf)
describes spatial and streaming prefetch mechanisms that can fetch more than
the explicitly requested line. Fences order operations; they do not disable
hardware prefetchers. The experiment does not modify BIOS or MSR settings.

### Per-group and per-row reports

Every new run also writes:

- `report.md`: detected 32-column ranges, counts and identities of detecting
  rows, spatial halo rates around injected groups, and a probe-order diagnostic.
- `analysis.json`: **all** groups, including undetected ones, with each row's
  hit count, repeat count, fraction, and corresponding control measurements.
- `row_sections.csv`: the same exhaustive per-case/per-row counts for plotting.

`lines.csv` now appends `sweep`, `probe_index`, and `previous_probe_line`.
`previous_probe_line` is the absolute cache-line index in the activation matrix,
not a feature column or virtual address. An empty field means the first probe
in that window. The report distinguishes spatial adjacency (higher feature
indices in the same row) from temporal adjacency (which line was probed before).

Three row descriptions are kept distinct:

- **Any-hit rows**: at least one repeat hit, useful for inspecting noise.
- **Consistent rows**: at least `--row-hit-fraction` of repeats hit (default 0.6).
- **Control-adjusted rows**: consistent and their gather hit fraction exceeds
  the larger of the two control hit fractions by `--min-control-delta` (default 0.5).

A group is detected when at least `--min-row-fraction` of its observed rows
are consistent (default 0.5). Control-adjusted detections apply the same rule
to adjusted rows. These rules use measured fractions, never injected labels.
Original per-line metrics and pass/fail criteria remain unchanged by parsing.

Stricter persistence may suppress stochastic neighboring hits:

```bash
python -m prefill_flush_reload --probe-rows 4 --row-hit-fraction .8 \
    --min-row-fraction .75 --output /tmp/prefill-four-rows-strict
```

This can also lose true detections; few rows/repeats provide weak evidence for
persistence. A repeatable gather-associated halo survives subtraction of the
no-gather controls. The analyzer reports contiguous detected runs as ambiguous
clusters instead of assuming the first group is genuine and deleting its
neighbors, since real outlier features can also be adjacent.

### Parse existing measurements without rerunning

The analyzer accepts the previous CSV schema too:

```bash
python -m prefill_flush_reload.analyze prefill_flush_reload/prefill-results \
    --row-hit-fraction .8 --min-row-fraction .75 --output /tmp/parsed-prefill

python -m prefill_flush_reload.analyze prefill_flush_reload/prefill-results \
    --probe-rows 4 --row-start 0 --output /tmp/parsed-first-four
```

These commands write analysis artifacts without changing the saved raw CSV
or its original summary. Parsing only four rows from an old full scan does
not undo reload interference during its original acquisition. Existing CSVs
have no probe-order metadata, so only the spatial diagnostic is available.

Deterministic row-selection and report tests:

```bash
python -m unittest prefill_flush_reload.test_simulation prefill_flush_reload.test_reporting -v
```

## Leading-group guesses and accuracy against the true groups

Reports now apply a forward-prefetch heuristic **after** the consensus step:
within each strictly consecutive run of candidate groups, keep the first
(lowest-column) group as the guess. For example, candidates at starts
`896, 928, 960, 1984` produce guesses `896, 1984`. The run beginning at 896
retains `928, 960` as following candidates, rather than deleting their evidence.
A missing group breaks a run; the heuristic does not bridge gaps.

The first section of `report.md` presents these guesses and their following
candidates. It scores both raw and processed predictions for consensus and
control-adjusted detection separately. Metrics include TP/FP/FN, precision,
recall, F1, overall group accuracy, Jaccard overlap, and exact-set equality
(Jaccard is in JSON/CSV). Prefer precision/recall/F1 and exact-set equality to
overall accuracy when almost all groups are negative.

These are **32-column group** metrics. A guess at 896 means columns 896–927;
it does not identify exactly which feature inside that cache line was an
outlier. Multiple real outlier columns inside one group count as one true
group. Ground truth only scores guesses; it never picks the first group.

This rule assumes that later neighbors are a forward halo. If two real outlier
groups are consecutive, the second will be suppressed and counted as a false
negative. Likewise, noise immediately before a true group can become a wrong
guess while suppressing the true group. Reports explicitly list real groups
among the following candidates so these failures remain visible.

Reprocess a saved run with the existing command:

```bash
python -m prefill_flush_reload.analyze prefill_flush_reload/prefill-results \
    --output /tmp/prefill-guesses
```

Additional machine-readable results:

- `analysis.json`: `guessed_groups`, `control_adjusted_guessed_groups`, and
  `group_detection.methods.{consensus,control_adjusted}`. Each method contains
  `raw` scores, `leading_group` scores, and `candidate_runs` retaining all
  following hits. Each group's per-row evidence is still in `groups`.
- `summary.json` from a new measurement: the same scores under
  `detection_summary.group_detection`.
- `group_scores.csv`: four records per run (two selectors, raw/processed), with
  observed row count and IDs, detection thresholds, precision/recall/F1,
  accuracy, exact-set match, and the predicted and true group lists.

A future row-count sweep can run fresh measurements for each `--probe-rows`
value and collect these CSV records. Keep full prefill size, seed, probe mode,
row-selection policy, and detection thresholds fixed when comparing row counts;
repeat across different seeds and row windows to assess reliability. Merely
filtering an existing full-scan CSV estimates parsing performance on subsets;
it does not recreate acquisition with fewer probes. Raw timing pass/fail status
remains independent of the processed group scores, so an otherwise noisy scan
can still produce a correct set of heuristic guesses.

Run semantic and scoring tests:

```bash
python -m unittest prefill_flush_reload.test_simulation \
    prefill_flush_reload.test_reporting prefill_flush_reload.test_scoring -v
```

## Paper figures from saved results

Install the optional plotting dependency in an environment of your choice:

```bash
python -m pip install -r prefill_flush_reload/requirements-plots.txt
python -m prefill_flush_reload.plots prefill_flush_reload \
    --output research_figures/prefill
```

For the temporary plotting environment prepared in this workspace session:

```bash
/tmp/bnb-paper-plots/bin/python -m prefill_flush_reload.plots prefill_flush_reload \
    --output research_figures/prefill
```

The program recursively discovers measurement `summary.json` files. You can
also pass several result folders, or the root of a future sweep. Overlapping
inputs are deduplicated. It reads existing analysis rules; older results with
no `analysis.json` are reconstructed from their raw CSV using recorded rules
or the analyzer defaults. This reconstruction does not modify the input files.
Inconsistent injected-group labels or incomplete per-row evidence are reported
as exclusions, rather than silently plotted with incorrect truth labels.

Exports default to **PDF, SVG, and 600-DPI PNG**. PDF/SVG location strips and
curves use vector elements; row heatmaps remain sampled images. PDF embeds
TrueType fonts and SVG retains editable text. Colors distinguish injected
groups (orange), raw candidates (blue), and first-group guesses (green), with
separate labeled rows. Curve markers and line styles distinguish predictions.

```bash
python -m prefill_flush_reload.plots prefill_flush_reload/prefill-16-rows \
    --formats pdf,svg --width 3.5 --output /tmp/one-paper-panel

python -m prefill_flush_reload.plots prefill_flush_reload \
    --selector control_adjusted --output /tmp/control-adjusted-figures
```

Generated figures and data:

- `overview-*.pdf/svg/png`: injected groups, raw candidates, and first-group
  guesses for up to eight acquisitions per page; each acquisition is labeled
  with its observed-row count, seed, row start, scan protocol, and flush scope.
- `runs/*/groups.*`: one acquisition's group comparison. Black tick marks
  indicate the exact injected feature indices; detections still resolve only
  32-column groups.
- `runs/*/rows.*`: token-row-by-column heatmaps of measured hit fractions across
  repeats, with a fixed 0–1 color scale. Orange outlines mark true injected
  groups, preserving visible evidence of neighboring hits and sporadic noise.
- `accuracy-cohort-*`: precision, recall, and F1 versus observed row count for
  matching experimental configurations. Both candidate and heuristic scores
  are shown.
- `plotted_groups.csv`: every plotted group's source, observed-row count,
  seed, injected/candidate/guess flags, and row-averaged hit fraction.
- `runs/*/hit_fractions.csv`: exact row-by-group values behind each heatmap.
- `performance.csv`: per-acquisition candidate and guess confusion counts,
  precision, recall, F1, and exact-set match, with protocol settings.
- `manifest.json`: all source paths, full cohort settings, reconstruction
  provenance, injected indices, scores, exported figure paths, and Matplotlib
  version. `README.md` in the figure directory explains the plots and exclusions.

A performance curve keeps the full tensor dimensions, **seed**, observed-row
start, probe mode/stride, flush scope, activation threshold, outlier count,
repeat count, and detection thresholds fixed. Changing any of these creates a
separate cohort. Unrecorded settings do not silently match recorded settings.
Thus seed-42 row-count runs are not connected to seed-420 runs, and a full-flush
64-row measurement is not connected to an observed-row-only flush curve.
A cohort containing only one count has location/heatmap figures but no curve.

If multiple acquisitions exist at one count in the same cohort, the curve
shows their mean and min/max range. These are descriptive ranges, **not
confidence intervals**. Single acquisitions have no invented error bars.
Undefined precision/recall (e.g. no guesses) is plotted as zero and preserved
as undefined in the machine-readable scores. No plot establishes a general
minimum row requirement from one seed or one acquisition per row count.

Suggested location-figure caption: “Injected outlier groups, measured candidate
groups, and first-group heuristic predictions for a synthetic prefill FFN
activation matrix. Each group spans 32 FP16 features. Black ticks denote exact
injected feature positions; predictions are evaluated at group resolution.”
State the chosen acquisition and analysis settings from `manifest.json` in
the paper's experimental setup. For heatmaps, note that hit fractions are
repeated cache observations of the same synthetic activation matrix.

Data-loading and cohort-integrity tests do not require Matplotlib:

```bash
python -m unittest prefill_flush_reload.test_plot_data -v
```
