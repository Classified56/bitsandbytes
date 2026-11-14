# Decode token classification and cache fidelity

Experiments collected on 2026-11-14 for the paper's decoding appendix. They
extend direct-channel classification and side-channel fidelity to generated
tokens from Qwen2.5-0.5B and Llama-3.2-1B-Instruct on Intel and AMD CPUs.

- [FINAL_RESULTS.txt](FINAL_RESULTS.txt): final numerical tables.
- [DIAGNOSTICS.txt](DIAGNOSTICS.txt): confidence intervals and additional checks.
- [paper/7-decoding-appendix.tex](paper/7-decoding-appendix.tex): the appendix
  added to Overleaf, including its five tables.
- [Release artifacts](https://github.com/Classified56/bitsandbytes/releases/tag/decode-token-recovery-20261114):
  all 62,946 binary observation/prediction files and all eight trained token
  classifiers. The two compressed downloads total about 1.31 GB.

## Main results

| Measurement | Qwen | Llama |
|---|---:|---:|
| Direct token accuracy, group 32, 1,536 test tokens | 74.93% | 65.89% |
| Intel cache-based token accuracy, 25 gathers, 128 tokens | 63.28% | 59.38% |
| AMD cache-based token accuracy, 25 gathers, 128 tokens | 65.62% | 61.72% |
| Exact 32-token prefixes, direct classification | 0/48 | 0/48 |
| Exact 8-token prefixes, cache-based classification | 0/16 | 0/16 |

The direct and cache-based rows above have different test sets. Use the paired
table in `FINAL_RESULTS.txt` to compare classification loss on the same tokens.
With 25 gathers, that loss is 0.78--3.12 percentage points. With one gather,
cache-based accuracy is much lower (3.91--33.59%).

## Method

| Setting | Value |
|---|---|
| CPUs | Intel Core i7-10700; AMD Ryzen 7 8700G |
| Models | Qwen2.5-0.5B base; Llama-3.2-1B-Instruct |
| Main computation | CPU INT8 products and FP16 outlier gathering/correction |
| Outlier rule | Frozen per-projection P99.9 threshold from training activations |
| Calibration | First 16 training-prompt prefills of the unquantized FP16 model |
| Features | Selected columns in all FFN `down_proj` layers; groups 1/4/16/32 |
| Direct split | 128 training, 24 validation, 48 test topics; 32 consumed generated tokens each |
| Classifier | One hidden layer, 1,024 ReLU units; AdamW; best validation checkpoint |
| Cache cohort | 16 test responses; prefill and eight decoding steps |
| Cache groups | 32 FP16 columns per 64-byte line |
| Gather conditions | One full scan; five full scans; stride-5 scans with 25 gathers |
| Cache timing | Original fenced RDTSC/RDTSCP implementation and scan scheduler |
| Postprocessing | Repeat consensus, row consensus, then consecutive-group filtering |

`PROTOCOL.json` and `BRIDGE_PROTOCOL.json` preserve the fixed experimental
choices. The label is the generated token **consumed** by the measured forward,
not its next predicted token. Qwen uses plain prompts; Llama uses its official
chat template. Unseen test token types count as classification errors.

The paper does not fully specify its top-k/robust-z selector. This experiment
reconstructs its numerically specified 99.9% baseline and states the percentile
population explicitly. It is a separate setting from the earlier threshold-6
experiments. Thresholds are frozen before acquisition and shared across CPUs.

All cache probes are synchronized inside the model process on experiment-owned
storage. The actual gathered values enter the correction. Every measured
response matches its uninstrumented run in logits and generated token IDs.
The 25-gather setting changes both scan order and repetition count. The
one-gather setting is one condition in a run that also tests other conditions
on the same activation. These are not measurements of an independently running,
unmodified victim or of NPU execution.

The direct classifiers are trained and evaluated on AMD. Paired classification
uses the frozen group-32 classifier without cache-specific retraining; offline
classification runs on Intel for both trace sources. Platform labels identify
the CPU that collected the traces. CPU arithmetic changes some Llama outputs,
so each trace is scored against its own generated tokens.

No-gather controls show that the scan itself contributes false positives.
Fixed-column controls separate row count from the selected columns. Real
prefill accesses are much denser under this threshold setting; filtering removes
many true adjacent groups. These prefill results do not repeat the paper's
original sparse setting.

## Files

| Path | Contents |
|---|---|
| `code/` | Acquisition, classification, analysis, verification, reporting, plotting, and artifact downloader |
| `profiles/`, `topics.json` | Frozen thresholds, calibration evidence, and fixed prompt topics |
| `results/amd/direct-*` | Direct features and generated-token records |
| `results/amd/classifiers-*` | Training histories, frozen classifier hashes, test predictions, and metrics |
| `results/{intel,amd}/live-*` | Raw timing records, selected columns, calibration, and forward validation |
| `results/{intel,amd}/controls` | No-gather and fixed-column controls |
| `results/{intel,amd}/bridge-*` | Paired direct/consensus/filtered token predictions |
| `runtime/`, `vendor/` | Exact CPU backend, original probe source, licenses, and runtime wrapper |
| `environment-*.json`, `*-provenance.json` | Software versions and source/model hashes |
| `logs/`, `pilot/`, `diagnostics/` | Execution records, pilots, and superseded runs excluded from final results |
| `figures/`, `paper/` | PNG/PDF plots and LaTeX tables/appendix |
| `results.json`, `verification.json` | Machine-readable results and independent verification |
| `SHA256SUMS` | Checksums for the repository files |
| `ARTIFACTS.json`, `ARTIFACT_SHA256SUMS` | Release locations, archive hashes, and every restored binary file hash |
| `PUBLICATION.json`, `provenance/` | Packaging changes and original acquisition notes |

Binary `.npz` and `.pt` files are restored into the paths above by the download
script. The `.pt` files are trained token classifiers, not LLM checkpoints.
LLM weights and authentication credentials are not distributed.
Redundant decoded response strings are replaced by their SHA256 hashes to keep
published text English-only. Exact token IDs are preserved; numerical results
and classifier labels are unchanged.

The protocol and original notes record the acquisition stage. `PUBLICATION.json`
records the later appendix and GitHub packaging changes.

The first AMD controls/Qwen runs overlapped or followed a bulk artifact copy.
They were assigned identical-settings quiet repeats before examining their
cache metrics. The quiet repeats are primary; the earlier records remain in
`diagnostics/`. Llama acquisition started after the transfer finished.

## Check the published results

Run from this folder. Only Python and NumPy are needed to verify stored records;
no model checkpoint, GPU, or NPU is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.6
python code/manifest.py --verify
python code/download_artifacts.py
python code/download_artifacts.py --verify-only
python code/verify.py
python code/report.py
```

Use Python 3.10 or newer. The downloader verifies the archive and each extracted
file, and refuses to overwrite changed files. For offline use, download both
Release assets first, then run
`python code/download_artifacts.py --from-dir /path/to/downloads`.

To recompute fidelity and confidence intervals from all primary raw traces:

```bash
python code/analyze.py results/intel/controls results/intel/live-qwen results/intel/live-llama \
  results/amd/controls results/amd/live-qwen results/amd/live-llama
python code/verify.py
python code/report.py
```

`report.py` reads the archived JSON results and does not retrain classifiers.
The numerical tables include all three gather conditions. The original report
generator is retained in `provenance/report-original.py`. Supplementary plots
can be rebuilt with Matplotlib using `python code/plot.py`.

## Repeat acquisition

Use Linux x86-64 with F16C/RDTSCP, a C compiler, and an authorized local copy of
each exact checkpoint revision in `PROTOCOL.json`. Install the measured CPU
packages in an isolated environment:

```bash
python -m pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r runtime/requirements-cpu.txt
```

Work in a separate copy and use new output paths. `runtime/run.sh` loads the
bundled backend. It does not require the original node's absolute paths.
For each model, set `MODEL` to `qwen` or `llama`, and `CHECKPOINT` to its local
directory. For example, on AMD:

```bash
MODEL=qwen
CHECKPOINT=/path/to/Qwen2.5-0.5B
mkdir -p rerun/profiles rerun/amd
bash runtime/run.sh python code/calibrate_model.py --model "$MODEL" \
  --checkpoint "$CHECKPOINT" --output "rerun/profiles/$MODEL.json"
bash runtime/run.sh python code/direct.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --profile "rerun/profiles/$MODEL.json" --output "rerun/amd/direct-$MODEL"
bash runtime/run.sh python code/classification.py --data "rerun/amd/direct-$MODEL" \
  --output "rerun/amd/classifiers-$MODEL" --stage train
bash runtime/run.sh python code/classification.py --data "rerun/amd/direct-$MODEL" \
  --output "rerun/amd/classifiers-$MODEL" --stage evaluate
```

Reuse the same newly frozen profile on both CPUs. On each CPU, set `PLATFORM`
to `intel` or `amd`; run controls once, then live acquisition for each model:

```bash
PLATFORM=intel
mkdir -p "rerun/$PLATFORM"
bash runtime/run.sh python code/controls.py --output "rerun/$PLATFORM/controls"
bash runtime/run.sh python code/live.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --profile "rerun/profiles/$MODEL.json" --output "rerun/$PLATFORM/live-$MODEL"
python code/analyze.py "rerun/$PLATFORM/controls" "rerun/$PLATFORM/live-$MODEL"
```

Run acquisitions sequentially, without concurrent classifier training or bulk
transfers. Transfer the completed direct/classifier and trace folders to Intel
for paired classification, preserving their relative paths:

```bash
bash runtime/run.sh python code/bridge.py --direct "rerun/amd/direct-$MODEL" \
  --classifiers "rerun/amd/classifiers-$MODEL" --live "rerun/$PLATFORM/live-$MODEL" \
  --output "rerun/$PLATFORM/bridge-$MODEL"
```

The original node launchers remain for provenance. Fresh measurements may
differ with CPU, runtime, scheduling, or numerical changes; the published
checksums identify the collected reference results, not an expected checksum
for a new run.
