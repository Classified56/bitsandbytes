#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPU_VENV="${NPU_VENV:-$HOME/npu-activity-experiment/cnn-venv}"
XRT_ROOT="${XRT_ROOT:-/opt/xilinx/xrt}"
source "$NPU_VENV/bin/activate"
source "$XRT_ROOT/setup.sh"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export MLIR_AIE_INSTALL_DIR="$SITE_PACKAGES/mlir_aie"
export PEANO_INSTALL_DIR="$SITE_PACKAGES/llvm-aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/python-deps:$ROOT/vendor:$ROOT/code:${PYTHONPATH:-}"
export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
exec "$@"
