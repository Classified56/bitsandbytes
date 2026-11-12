#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CPU_VENV:-}" ]]; then
    source "$CPU_VENV/bin/activate"
fi
export PYTHONPATH="$ROOT/vendor:$ROOT/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
exec "$@"
