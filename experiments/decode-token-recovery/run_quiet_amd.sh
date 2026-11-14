#!/usr/bin/env bash
# Prespecified quality repeat after an artifact-copy overlap, before metrics.
set -euo pipefail
cd "$(dirname "$0")"
test "$(cat logs/fidelity-amd.exit)" = 0
test ! -e diagnostics/amd-controls-before-quiet-repeat
test ! -e diagnostics/amd-qwen-during-artifact-copy
python3 - <<'PY'
import json
from pathlib import Path
for dataset in ['controls','live-qwen','live-llama']:
    assert json.loads((Path('results/amd')/dataset/'manifest.json').read_text())['complete']
PY
mv results/amd/controls diagnostics/amd-controls-before-quiet-repeat
mv results/amd/live-qwen diagnostics/amd-qwen-during-artifact-copy
mv logs/controls-amd.log diagnostics/controls-amd-before-quiet-repeat.log
mv logs/live-amd-qwen.log diagnostics/live-amd-qwen-during-artifact-copy.log
runtime=/home/tianhong/qwen-performance-20261111/run_amd.sh
bash "$runtime" python code/controls.py --output results/amd/controls >logs/controls-amd.log 2>&1
bash "$runtime" python code/live.py --model qwen --checkpoint /home/tianhong/qwen-performance-20261111/checkpoint --profile profiles/qwen.json --output results/amd/live-qwen >logs/live-amd-qwen.log 2>&1
echo 0 >logs/quiet-amd.exit
