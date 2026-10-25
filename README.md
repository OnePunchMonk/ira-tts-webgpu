# Ira · text-to-speech in the browser (WebGPU)

A 2-bit build of [rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) (voice **Ira**; Hindi / English / Hinglish) that runs entirely on the
user's GPU through WebGPU, plus the pipeline that produced it. Research / non-commercial use only (see *Licence*).

* Model + weights (public): **https://huggingface.co/Sanyam0605/ira-tts-2bit-web** — 1.2 GB decoder (≈2.6 effective bits) + Mimi decoder + tokenizer assets.
* Demo page: `web/` (static; loads the weights from the Hugging Face repo).

## Status — honest summary
| | result |
|---|---|
| 2-bit QAT quality (Whisper-large-v3-turbo, 160 held-out sentences, WER capped at 100 %) | **23.8 %** vs ≈22 % for the original model (English 9.6 % vs ≈9 %) |
| Failure rate | ≈1 repetition loop in 160 generations ("sigh sigh sigh…") |
| Runs in a real browser | **Yes** — Apple Metal-3, Chrome/WebGPU accepts the 2-bit `MatMulNBits` weights |
| Speed (first measurement) | **8.6 tokens/s**; real time needs ≥ 100 → ~12× too slow. Work in progress: fused `GroupQueryAttention` graph (≈450 ops instead of ≈5,500), see `pipeline/web_tts/build_fast_graph.py` |
| Expression tags `<laugh>/<chuckle>/<sigh>` | mostly English only — same in the original model |

What did **not** work (kept here because it is useful to know): distilling the teacher into a pruned 12-layer / 494 M student with ~11 M tokens of teacher data
(`train_student.py`) gave unintelligible speech (WER > 100 %; text barely steers it). Post-training quantization of the teacher breaks between 3.0 and 3.3 effective bits;
quantization-aware fine-tuning (`qat_teacher.py`) is what makes 2-bit work.

## Repo map
```
web/                     the browser app (index.html, app.js, lib.js, serve.py, setup_vendor.sh)
pipeline/
  student_data/          corpus (3,019 + 5,935 Hindi/English/Hinglish lines, hand-placed expression tags), teacher-data generation (vLLM),
                         student distillation (failed), quantization sweeps, sensitivity scan, QAT, evaluation scripts, logs/
  web_tts/               ONNX export: ref_decoder.py (exact Cohere2 re-implementation, parity-tested), export/quantize/verify, fused-op graph builder, Mimi 8-codebook patch
  vllm-plugin/           vLLM plugin for the TinyAya/Cohere2 + stop-head architecture (NVFP4 teacher on DGX Spark)
  docs/                  prompt format and controls of rumik-oss-1, notes
samples/                 audio (Git LFS): teacher reference, teacher vs 2-bit pairs, early student failures
```
The pipeline scripts assume the workspace layout of the original project: `pipeline/` is the workspace root, with the downloaded model directories
(`rumik-oss-1/`, `rumik-oss-1-nvfp4-vllm/`) next to `student_data/`. Weights, teacher dumps (`.npz`), raw datasets and ONNX files are not in git.

## Run the demo
```bash
cd web && ./setup_vendor.sh        # fetches onnxruntime-web 1.30 + @huggingface/tokenizers into web/vendor (needs node/npm)
python3 serve.py                   # http://localhost:8000 in Chrome 113+ (WebGPU)
```
Click **Load model** (≈1.4 GB from Hugging Face, cached by the browser), **Speed test**, **Speak**. Add `?profile=1` for a GPU kernel breakdown in the Log.

## Licence
Code: no licence chosen yet. Model weights are derived from rumik-oss-1 (built on CohereLabs/tiny-aya-fire) and are **CC BY-NC 4.0 with the Cohere Labs acceptable-use
addendum**: research and non-commercial use only. Mimi codec: Kyutai, CC BY 4.0. Corpus text: written for this project (prompts drawn from sarvamai/samvaad-hi-v1, Apache-2.0).
