#!/usr/bin/env bash
# Fetch the two JS libraries the page needs into web/vendor/ (not committed: ~83 MB of wasm).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf _tmp && mkdir -p _tmp vendor/ort vendor/tokenizers
(cd _tmp && npm init -y >/dev/null && npm install --silent onnxruntime-web@1.30.0 @huggingface/tokenizers@0.2.0)
cp _tmp/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs _tmp/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.mjs _tmp/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.wasm vendor/ort/
cp _tmp/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs vendor/tokenizers/
rm -rf _tmp
echo "vendor ready: $(du -sh vendor | cut -f1)"
