#!/usr/bin/env bash
# Run the teacher pass in the vLLM docker image. Extra args go to gen_teacher.py.
# Smoke test:  ./student_data/run_teacher.sh --limit 24 --save-wavs 24 --out-dir /workspace/student_data/teacher_smoke
# Full run:    ./student_data/run_teacher.sh
# Requires the GB10 clock cap first:  sudo nvidia-smi --lock-gpu-clocks=0,1500
set -euo pipefail
cd "$(dirname "$0")/.."
# rumik-oss-1-nvfp4-vllm is a symlink into the HF cache whose files are ABSOLUTE symlinks to <root>/blobs/..., so
# mount the whole HF model root at the same absolute path inside the container or the files dangle.
SNAP="$(readlink -f rumik-oss-1-nvfp4-vllm)"
ROOT="$(dirname "$(dirname "$SNAP")")"
docker run --rm --gpus all \
  -v "$ROOT":"$ROOT":ro \
  -v "$PWD/vllm-plugin":/workspace/vllm-plugin \
  -v "$PWD/student_data":/workspace/student_data \
  --entrypoint bash vllm/vllm-openai:v0.20.0 \
  -c "pip install -q -e /workspace/vllm-plugin && VLLM_PLUGINS=register_tinyaya python3 -u /workspace/student_data/gen_teacher.py --repo-dir $SNAP $*"
