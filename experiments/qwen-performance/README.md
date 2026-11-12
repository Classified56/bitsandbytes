# Qwen2.5-0.5B CPU and AMD NPU performance

Performance measurements collected on 2026-11-11 for the paper's architecture
performance table. This experiment measures time to first output token (TTFT)
and memory for six configurations. The timed workload contains no attack code,
cache probes, or concurrent observer workload.

See [FINAL_RESULTS.txt](FINAL_RESULTS.txt) for the concise performance tables.
The numbers describe these implementations, not maximum hardware performance.

## Protocol

| Setting | Value |
|---|---|
| Model | Official `Qwen/Qwen2.5-0.5B` base model |
| Model revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Paper bitsandbytes source | `c593d6f9debc17253c6b74fd6c2c1dbceecc4fcd` |
| Input / batch / output | 128 tokens / 1 / first output token |
| Requests per configuration | 2 warmups, then 10 measured requests |
| CPU allocation | 4 distinct physical cores, 4 Torch threads, 1 interop thread |
| Attention | Hugging Face eager attention; fresh KV cache per request |
| Prompt | `prompt_1` in `prompts.json`, identical across all six runs |
| Reported latency | Median TTFT; minimum and maximum also retained |
| DQ threshold | 6.0 |
| Converted projections | All 168 transformer linear layers across 24 decoder blocks |

TTFT starts before tokenization of the raw prompt and ends after the first
greedy output token is decoded to text. It includes one batched 128-token model
forward pass, quantization, transfers, synchronous dispatches, and CPU correction
where applicable. It excludes model loading, NPU JIT compilation, warmup, queueing,
and network transport. Prefix-cache reuse is disabled. This is a local inference
benchmark, not a serving throughput or multi-token decode benchmark.

After model conversion, every configuration runs `gc.collect()` and
`malloc_trim(0)` once, before warmup, to release dead preparation allocations.
No trimming occurs during the timed requests. Peak memory is the maximum host
RSS sampled by a separate parent process during the measured phase, plus the
mapped XRT allocation when using the NPU. Sampling targets 5 ms; actual sampling
interval statistics are recorded in each `summary.json`. Shorter memory spikes
can be missed. Loading and compilation peaks are excluded.

The XRT mappings are shared DDR allocations, not additional NPU SRAM. Their
`/proc/PID/smaps` RSS is zero on this driver, so adding them avoids omitting NPU
buffers without double-counting host RSS. Mapping size measures allocated/mapped
memory; physical residency was not measured separately. The total includes the
mapped XRT runtime pool as well as tensors. Tensor-only allocation is also in
`summary.json`. All measured workers had zero swap usage.

## Execution paths and precision

| Path | Main linear product | Outlier correction |
|---|---|---|
| CPU without DQ | FP32 CPU | None |
| CPU with DQ | INT8 quantization; platform backend below | FP32 CPU |
| CPU+NPU without DQ | BF16 input/weights, FP32 accumulation/output | None |
| CPU+NPU with DQ | INT8 input/weights, INT32 accumulation | FP32 CPU |

CPU FP32 is the paper repository's unquantized baseline. The BF16 NPU baseline
is an available hardware execution path, so the CPU/NPU comparison is not a
matched-precision hardware speed comparison.

DQ follows the pinned bitsandbytes mixed INT8 operation: FP16 quantizer input,
per-row dynamic activation scaling, whole-column outlier selection at 6.0,
suppression of selected columns in the INT8 product, and CPU gather/dequantize/
`addmm` correction. The corrected weights are dequantized columns of the retained
INT8 weights, as in that repository's inference implementation. The wrapper is
numerically checked against its actual `int8_mixed_scaled_mm` operator.

The Intel AVX2 backend falls back to FP32 matrix multiplication for the INT8
product. The AMD AVX512 backend uses `torch._int_mm`. This backend difference is
part of the observed latency result, not evidence of a universal DQ speedup.

The hybrid paths issue **168 real synchronous NPU matmuls per request**, with no
CPU matmul fallback. They use **one AIE compute core**, not the full NPU array.
Weights persist in XRT buffers and shape-specific input/output buffers are
reused. Attention, normalization, nonlinear operations, embeddings, the final LM
head, and DQ quantization/dequantization/correction remain on the CPU. The DQ CPU
correction reads the same shared-DDR INT8 weights as the NPU. The current hybrid
implementation is slower than the CPU baselines and includes substantial host,
transfer, and dispatch overhead.

## Platforms

| Item | Intel | AMD |
|---|---|---|
| Processor | Core i7-10700 | Ryzen 7 8700G |
| NPU used | None | XDNA 1, RyzenAI-npu1 |
| Python | 3.10.12 | 3.12.3 |
| Kernel | 6.8.0-107-generic | 7.0.0-42-generic |
| Torch | 2.11.0+cpu | 2.11.0+cpu |
| Transformers | 4.57.1 | 4.57.1 |
| NumPy / Tokenizers / Safetensors | 2.2.6 / 0.22.2 / 0.8.0 | Same |
| MLIR-AIE | Not used | 1.4.4 |
| Peano (`llvm-aie`) | Not used | 22.0.0.2026092101+0006955e |
| XRT / NPU firmware | Not used | 2.21.75 / 1.5.5.391 |

The core model packages are pinned equally; Python, operating system, and CPU
backends differ as recorded. No CPU governor, clocks, or driver settings were
changed. These are short single-session measurements on the available nodes,
not controlled thermal/power or cross-day reproducibility experiments.

## Files

| Path | Contents |
|---|---|
| `FINAL_RESULTS.txt` | Performance-only final tables |
| `code/` | Exact final benchmark worker, sampler, DQ wrapper, NPU kernel, validation, summary generator |
| `prompts.json` | Three fixed 128-token prompts and exact token IDs; only the first is timed for the final table |
| `results/<configuration>/worker.json` | Every warmup, timed request, validation, settings, and software versions |
| `results/<configuration>/rss-samples.json` | Raw timestamped RSS samples, including setup and measured phases |
| `results/<configuration>/worker.log` | Worker output and exact measured-phase boundaries |
| `results/<configuration>/smaps-ready.txt` | Memory mappings after warmup |
| `results/<configuration>/validation-logits.npz` | Full first-token logits for three prompts |
| `results/kernel-validation.json` | NPU products and DQ operator checks |
| `results/cross-validation.json` | Cross-configuration logits comparisons |
| `provenance/` | Pinned source archive, model hashes, software environment, original launchers, kernel patch, license |
| `pilot/` | Earlier untrimmed conversion runs and failed wide-kernel attempt; excluded from final results |
| `paper/` | Performance table and corresponding prose used in Overleaf |
| `SHA256SUMS` | Checksums of the published files, excluding the manifest itself |

The pilot's original four-row DMA schedule exceeded XDNA 1 shim descriptor limits
for a wide Qwen FFN matrix. `single-core-dma.patch` reduces the schedule block
size to two rows. All four actual Qwen matrix shapes were revalidated after this
change. The final six configurations were then measured again with the consistent
allocator policy above; no pilot measurements enter the final table.

The new batched prefill protocol also differs from the earlier separate custom
NPU engine's sequential-token experiment. Do not combine those earlier timings
with this six-way comparison. Rocket+Gemmini was not measured and remains `TBD`
in the paper table.

## Reproduce

Use Linux on each target machine. Work in a copy of this folder and move the
bundled `results/` to `reference-results/` before running, so new measurements do
not overwrite the published evidence. The benchmark requires a local checkpoint
and the pinned bitsandbytes Python source; it does not require CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
mkdir -p vendor results
tar -xf provenance/bitsandbytes-source.tar -C vendor
python code/download_model.py
```

The downloader uses the pinned official Hugging Face revision and verifies file
sizes and SHA256 hashes. Model weights and installed packages are not committed.
`vendor/` supplies the pinned operators; do not silently substitute a different
installed bitsandbytes release.

On Intel, activate the CPU environment (or set `CPU_VENV`), then run:

```bash
bash run_intel.sh python code/run_benchmark.py --label intel_cpu_no_dq --backend cpu --dq 0
bash run_intel.sh python code/run_benchmark.py --label intel_cpu_dq --backend cpu --dq 1
```

For AMD, use a functioning XDNA 1 Linux driver, XRT, and the MLIR-AIE/Peano
environment listed above. Set `NPU_VENV` to that Python 3.12 environment and
`XRT_ROOT` if XRT is not installed at `/opt/xilinx/xrt`. Driver/toolchain installation
is a prerequisite, not performed by these scripts. Official toolchain sources:
[MLIR-AIE](https://github.com/Xilinx/mlir-aie/releases/tag/v1.4.4) and
[XDNA driver](https://github.com/amd/xdna-driver).

The measured AMD node used an existing NPU environment with the same pinned
Torch/model packages installed into an isolated `python-deps/` overlay:

```bash
"$NPU_VENV/bin/python" -m pip install --target python-deps torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$NPU_VENV/bin/python" -m pip install --target python-deps -r requirements-cpu.txt
bash run_amd.sh python code/test_kernels.py
bash run_amd.sh python code/run_benchmark.py --label amd_cpu_no_dq --backend cpu --dq 0
bash run_amd.sh python code/run_benchmark.py --label amd_cpu_dq --backend cpu --dq 1
bash run_amd.sh python code/run_benchmark.py --label amd_npu_no_dq --backend npu --dq 0
bash run_amd.sh python code/run_benchmark.py --label amd_npu_dq --backend npu --dq 1
```

Run configurations sequentially on each node, with no competing benchmark.
Place the six result directories under one `results/` directory, then regenerate
the summary and paper table with:

```bash
python code/summarize_results.py
sha256sum -c SHA256SUMS  # Check the unmodified published bundle only.
```

The portable launchers preserve the measured thread and path setup while making
environment locations configurable. `provenance/measured-run_*.sh` preserves the
original node launchers. Full environment inventories include unused packages;
they document the host and are not intended as portable pip requirements.

## Numerical validation and scope

All four Qwen matmul shapes pass on the NPU: `[M,K,N]` = `[128,896,128]`,
`[128,896,896]`, `[128,896,4864]`, and `[128,4864,896]`. INT8 products match the CPU
integer reference exactly; BF16 products differ by at most 0.00049 in the test.
The wrapper matches the pinned mixed DQ operator with and without injected
outliers. All six configurations produce the same greedy first token for each of
three validation prompts. AMD CPU DQ and AMD NPU DQ full logits are identical on
all three. The raw logits and comparison metrics are retained for inspection.
These checks do not replace a model-quality, perplexity, or full-generation study.

The copied AMD kernel is licensed Apache-2.0 WITH LLVM-exception; see
`provenance/MLIR-AIE-LICENSE.txt`. The bitsandbytes source archive includes its
own license. The model's license is downloaded and hash-verified with the model.
