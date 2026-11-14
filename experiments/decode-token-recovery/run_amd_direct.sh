#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime=/home/tianhong/qwen-performance-20261111/run_amd.sh
mkdir -p results/amd logs
for model in qwen llama; do
  if [ "$model" = qwen ]; then checkpoint=/home/tianhong/qwen-performance-20261111/checkpoint; else checkpoint=/home/tianhong/llama-direct-decode-20261114/checkpoint; fi
  bash "$runtime" python code/direct.py --model "$model" --checkpoint "$checkpoint" --profile "profiles/$model.json" --output "results/amd/direct-$model" >"logs/direct-$model.log" 2>&1
  bash "$runtime" python code/classification.py --data "results/amd/direct-$model" --output "results/amd/classifiers-$model" --stage train >"logs/train-$model.log" 2>&1
  bash "$runtime" python code/classification.py --data "results/amd/direct-$model" --output "results/amd/classifiers-$model" --stage evaluate >"logs/evaluate-$model.log" 2>&1
done
