#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
platform=$1
if [ "$platform" = intel ]; then runtime=/home/tianhong/qwen-performance-20261111/run_intel.sh; else runtime=/home/tianhong/qwen-performance-20261111/run_amd.sh; fi
mkdir -p "results/$platform" logs
bash "$runtime" python code/controls.py --output "results/$platform/controls" >"logs/controls-$platform.log" 2>&1
for model in qwen llama; do
  if [ "$model" = qwen ]; then checkpoint=/home/tianhong/qwen-performance-20261111/checkpoint
  elif [ "$platform" = intel ]; then checkpoint="$PWD/checkpoints/llama"
  else checkpoint=/home/tianhong/llama-direct-decode-20261114/checkpoint; fi
  bash "$runtime" python code/live.py --model "$model" --checkpoint "$checkpoint" --profile "profiles/$model.json" --output "results/$platform/live-$model" >"logs/live-$platform-$model.log" 2>&1
done
