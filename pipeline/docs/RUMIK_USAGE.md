# rumik-oss-1: how to prompt it (from the HF card + code)

Source: `rumik-oss-1/README.md` (HF card), `server.py`, `tinyaya_manifest.json`.
This is the reference for generating teacher data for the distilled Ira student.

## Prompt format (exact)

The model sees one string; the tokenizer turns it into a token sequence and the model emits Mimi audio tokens.

```
<text>{Speaker}: {description?} {text with inline tags}<audio>
```

Built in `server.py`: `prompt = f'<text>{req.speaker}: {req.input}<audio>'`.
So the speaker is a plain-text prefix, not a separate embedding or id. `input` holds the optional description prefix plus the text.

Audio tokens look like `<value_quantizer>` (regex `^<(\d+)_(\d+)>$`), 8 per Mimi frame, 12.5 frames/s, 24 kHz output.
Generation ends on `</audio>`. The decoder only samples from audio tokens plus `</audio>`.

## Speakers

`Ira`, `Aisha`, `Siya`, `Zoya`. All four are meant to work in all 22 languages.
`tinyaya_manifest.json` also lists a fifth speaker, `Silver`, which the server rejects and the card does not document. Don't rely on it.

## Delivery controls

`<description="...">` goes at the start of the text. It sets the global tone, accent and pace.

```
<description="excited, Hindi accent, fast pace"> जल्दी आओ, हमारा नाम लिस्ट में है! <laugh> आज घर में जश्न होगा।
```

| control | values |
|---|---|
| tone | happy, sad, angry, excited, professional |
| accent | Hindi, Telugu, Tamil, Kannada, Bengali, Punjabi, Indian English |
| pace | slow, fast, steady |
| inline vocalization | `<laugh>`, `<chuckle>`, `<sigh>` (placed at the word position) |

Description is a comma-separated list: `tone, accent, pace`. The card examples always use all three, for example `happy, Hindi accent, steady pace` and `professional, steady pace`.

## Languages and code-switching

- 22 Indic languages in native script or romanized, plus English.
- Code-switching works within one utterance. Card example (Hindi + English): `आज की meeting में सबको हमारा idea पसंद आया। I was so nervous, लेकिन तुमने कहा था ना...`
- For our assistant (Hindi + English): use Devanagari for Hindi words and Latin for English words. Pair `Hindi accent` with Hindi-heavy text and `Indian English` with English-heavy text.

## Sampling settings

Card showcase: temperature `0.8`, top-k `30`, top-p `1.0`, max `3072` tokens. Server default: `max_new_tokens` 2048. Keep the same sampling for teacher data.

## Limits

- Training utterances are at most 30 s; the card advises against more than 35 s. Split long replies at sentence boundaries.
- Vocalization tags: unrequested vocalizations show up in 2.9-5.4% of outputs. Tag adherence was only measured in English (NoVA 0.884).
- Quality context: IndicEmo 2.92/5 overall; WER/CER per language is in the card benchmarks.

## Minimal call

```bash
python ./rumik-model/server.py --repo-id ./rumik-model --host 127.0.0.1 --port 6006

curl http://127.0.0.1:6006/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "speaker": "Ira",
  "input": "<description=\"happy, Hindi accent, steady pace\"> नमस्ते, आज आपका दिन कैसा रहा?",
  "temperature": 0.8, "top_k": 30, "max_new_tokens": 2048 }' --output speech.wav
```

## Licensing (matters for the browser release)

rumik-oss-1 is CC-BY-NC 4.0 with Cohere's acceptable-use addendum (via tiny-aya-fire). Research and non-commercial only; sharing adaptations is allowed with attribution, license links and a note of modifications. A distilled student trained on its outputs is an adaptation of it, so the student's public demo must stay non-commercial and carry the same attribution. Mimi is CC-BY-4.0 (separate).

## Plan: Ira-only student

"Cloning Ira" needs no cloning step, because Ira is a built-in speaker. We distill by using `Ira:` as the only speaker when generating teacher data, so the student is a single-voice model and needs no speaker conditioning.

1. Build a Hindi/English text corpus: assistant-style replies, code-switched, short (1-3 sentences, under 15 s), with a spread of tone/pace descriptions and a few vocalization tags.
2. Run the teacher (vLLM NVFP4 plugin, `vllm-plugin/`) with `Ira:` to produce text → Mimi tokens. Keep the teacher logits (top-k) for logit distillation.
3. Prune the vocab to audio tokens (16,384) plus the text tokens we use, then train a 300-500M student on these pairs.
4. Quantize to 4-bit, export to ONNX/WebGPU. Mimi decoder runs in the browser. LFM2 writes the reply text.
