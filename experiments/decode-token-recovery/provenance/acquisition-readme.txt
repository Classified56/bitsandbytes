DECODE EXTENSION: DIRECT CLASSIFICATION AND CACHE FIDELITY

This package extends sections V-A and V-B to generated-token decoding on
Qwen2.5-0.5B BASE and Llama-3.2-1B-Instruct, on the authorized AMD and Intel CPUs.
The old fixed-threshold-6 experiments are retained in their original folders.

METHOD RECONSTRUCTION
The current manuscript describes an offline per-layer 99.9% threshold baseline,
runtime column maxima, down_proj features, FP16 storage, and group widths
1/4/16/32. It also mentions top-k/robust-z without specifying k, a z cutoff,
calibration inputs, or a precise percentile population. The original selector
implementation is not present in the supplied GitHub branch.

PROTOCOL.json explicitly operationalizes the most numerically specified
baseline: per-projection scalar P99.9 of absolute FP16 training activations.
This is a reconstruction from the text, not proof of bit-identical reproduction
of the original unpublished selector. All thresholds are calibrated from the
first 16 TRAIN prompt prefills of the unquantized checkpoint, then frozen before
mixed-INT8 inference. The same profile JSON is used on both CPUs. Q/K/V share
their input threshold, as do gate/up. No threshold is tuned using test accuracy.

All transformer projections execute INT8 main products plus FP16 gathered
corrections. Only down_proj inputs are measured for the primary results.
Activations really use FP16 storage: each 64-byte line spans 32 columns.
The final lm_head remains FP16. This differs from the preceding FP32-gather
fixed-threshold-6 experiment and must be labeled as a separate configuration.

V-A
The fixed 128/24/48 topic-disjoint response split supplies up to 32 consumed
generated tokens per response, stopping before EOS/EOT. Prompts are neutral
encyclopedia instructions. Llama uses its official chat template; Qwen uses
plain text because the manuscript's Qwen checkpoint is BASE. Test token types
absent from training count as errors. Train-only feature selection and best-dev
checkpoint selection precede all current test evaluation. The same topic split
was used in earlier experiments; this is not a fresh independent topic cohort.

V-B
The original GitHub native timing and observation scheduler is unchanged;
vendor-provenance.json records its commit and file hashes. Timings use the
paper's fenced RDTSC/RDTSCP primitive. The probe is synchronized in-process on
owned activation storage; gathered values feed the actual outlier correction.
Every measured response is rerun without probes and must produce bit-identical
logits and generated IDs. These observations do not establish an attack against
an unmodified victim, autonomous boundary discovery, or private-page sharing.

There are three prespecified regimes: full scan with 1 gather, full scan with 5
gathers, and stride5 with 5 repeats (25 gathers). Rows 1/4/16/32 are observed for
prefill; decode has one row. Different decode steps are never pooled as repeat
observations of the same footprint. Negative controls omit gather entirely.
Synthetic controls keep four selected columns fixed while changing rows 64/1.
Real prompts retain natural token lengths rather than inserting padding tokens.

Each model/platform calibrates its own cache timing cutoff using independent
primitive labels; training 1000 and validation 1000 samples. The cutoff stays
fixed across that model's phases/regimes. Quantization thresholds and cache-hit
timing cutoffs are different quantities. Neither uses test token labels.

Raw errors, repeat/row consensus, and consecutive-group filtering are reported
separately. Filtering can remove genuine adjacent positives. Spatially grouped
or repeated observations are not independent statistical samples; intervals
resample whole prompts or synthetic blocks. Single-acquisition data is reported
alongside replay fidelity. CPU main products are used throughout; no NPU result
is claimed here. "Single execution" means one gather inside that measurement
condition. Other conditions also run on the same activation in randomized order;
this is not an independently running, never-replayed victim service.

BRIDGE
BRIDGE_PROTOCOL.json applies the frozen direct-feature group32 classifier to
actual direct/consensus/filtered masks from the same predefined live test cohort.
It quantifies the direct-to-cache accuracy gap without cache-specific retraining.
This smaller cohort contains up to 128 positions (16 responses x 8 steps).
Its exact-prefix metric concerns eight tokens, not a complete generated reply.
All direct classifiers are trained and tested on AMD. Paired bridge inference
is performed offline on Intel for both trace sources, keeping the classifier
execution host fixed. Platform labels in bridge tables identify the host that
collected the cache traces, not the host performing offline classification.

REPRODUCTION
Use the existing platform wrapper in /home/tianhong/qwen-performance-20261111/:
run_amd.sh or run_intel.sh, with the checkpoint paths in the runner scripts.
1. Run code/calibrate_model.py on AMD for each model, before any acquisition.
2. Copy the identical profiles/ JSON files to Intel; run direct/live pilots.
3. Run run_amd_direct.sh on AMD. Train and freeze all widths before testing.
4. Run run_fidelity.sh PLATFORM sequentially on each platform, avoiding overlap
   with local classifier work. Keep controls and real traces on the same host.
5. Run code/analyze.py on completed controls/live directories.
6. Run code/bridge.py with matching direct, classifiers, live, and output paths.
7. Verify predictions and generate final tables from archived raw records.

PORTABLE RUNTIME
runtime/vendor/bitsandbytes contains the exact 47 Python backend files used by
the measured environments, including its license. Their hashes match
environment-intel.json and are checked against environment-amd.json. These CPU
operators use PyTorch; no CUDA or NPU extension is needed for this experiment.
Use Linux x86-64 with F16C/RDTSCP, a C compiler, Python3.10/3.12, CPU Torch2.11.0,
and the package versions in runtime/requirements-cpu.txt. The original node
wrappers are preserved in runtime/ for provenance. After activating an
equivalent environment, prefix commands with "bash runtime/run.sh python".
For collection, supply an authorized local --checkpoint directory and use a
new output folder. Classifier, analysis, bridge, and verification use no LLM
checkpoint. Verification and reporting from this archive:
  bash runtime/run.sh python code/verify.py
  bash runtime/run.sh python code/report.py
  python code/plot.py
  python code/manifest.py --verify
The plots require Matplotlib. They are supplementary exports, not inputs to
training, threshold selection, or evaluation.

FILES
PROTOCOL.json, BRIDGE_PROTOCOL.json: fixed choices and disclosed assumptions
profiles/: frozen training-only thresholds and calibration evidence
results/amd/direct-*, classifiers-*: V-A data, histories, predictions, checkpoints
results/{intel,amd}/controls, live-*: V-B timings, truth, calibration, validation
results/{intel,amd}/bridge-*: paired direct-to-cache token predictions
environment-*.json: runtime versions and backend source hashes
runtime/: exact backend source, original wrappers, portable wrapper, dependencies
logs/, diagnostics/, pilot/: audit records, including failed pre-measurement setup
FINAL_RESULTS.txt: concise numerical result tables
paper/: reviewable LaTeX tables; these do not modify the Overleaf project
figures/: standalone PNG/PDF exports from verified numerical results
SHA256SUMS: portable archive checksums, excluding model weights

Gated checkpoint weights stay on the explicitly authorized nodes and are not
included in the portable results archive. Authentication tokens are never copied.

ACQUISITION QUALITY REPEAT
The first AMD controls/Qwen runs were adjacent to or briefly overlapped an
archive transfer of completed classifier artifacts. Before examining AMD cache
metrics, both were assigned complete identical-settings repeats with no bulk
transfer or classifier work. These quiet repeats are primary regardless of
outcome. Original raw runs/logs are retained in diagnostics/. Llama acquisition
began after that transfer ended and is retained. run_quiet_amd.sh records the
repeat procedure; diagnostics/execution-notes.json records the decision.
