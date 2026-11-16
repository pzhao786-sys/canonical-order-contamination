#!/usr/bin/env bash
# Fetch the authors' data on the Jetstream2 GPU machine (run after scripts/remote_env.sh).
set -euo pipefail
source /workspace/env.sh

mkdir -p /workspace/data
cd /workspace/data
if [ ! -d test_set_contamination ]; then
  git clone --depth 1 https://github.com/tatsu-lab/test_set_contamination
fi
cd test_set_contamination/benchmarks
if [ ! -d unzipped ]; then
  python -m zipfile -e benchmarks.zip unzipped
fi

echo "=== canary files (authors' intentionally contaminated model) ==="
find /workspace/data/test_set_contamination/detection_challenge_benchmarks -type f | sort
echo "=== Table 2 benchmark files ==="
find /workspace/data/test_set_contamination/benchmarks/unzipped -type f | sort | head -n 200
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
