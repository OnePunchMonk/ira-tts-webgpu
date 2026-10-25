"""
Sanity-check the teacher data before training the student on it.

 1. Load teacher/ + teacher2/ chunks, drop unterminated samples, build manifest.jsonl
    (id, n_samp, src, lang, text, description, n_tok, secs, secs_per_word, flag).
 2. Flag duration outliers (audio seconds per word) - truncated / runaway / mumbled generations.
 3. Decode a stratified sample to wav (listen/) + listen/index.md.
 4. Whisper-large-v3-turbo ASR on a sample, WER/CER vs the input text (tags + description stripped).
    Whisper's Hindi/Hinglish transcripts are noisy, so use WER as a RELATIVE number (teacher vs student).

Run on the host .venv (needs GPU; do not run while a training job is on the GPU).
"""
import argparse
import glob
import json
import random
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
TEACHER_REPO = ROOT.parent / "rumik-oss-1"
AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")
TAG = re.compile(r"<(laugh|chuckle|sigh)>")
DESC = re.compile(r'^<description="[^"]*">\s*')
HZ = 12.5


def load_corpora():
    meta = {}
    for fn in ("corpus.jsonl", "corpus2.jsonl"):
        for l in open(ROOT / fn):
            d = json.loads(l)
            meta[d["id"]] = d
    return meta


def group_of(rid):
    return {"c": "conv", "s": "trivia", "l": "trivia", "t": "round2"}[rid[0]]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_text(t):
    t = DESC.sub("", t)
    t = TAG.sub(" ", t)
    t = unicodedata.normalize("NFC", t).lower()
    # drop only punctuation/symbols (Unicode P*/S*); \w would also strip Devanagari matras (category M*)
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wavs", type=int, default=60)
    ap.add_argument("--n-asr", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    meta = load_corpora()

    from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel, pipeline

    tok = AutoTokenizer.from_pretrained(TEACHER_REPO, trust_remote_code=True)
    vocab = tok.get_vocab()
    end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}

    # ---- 1/2: manifest + outliers (token ids kept in memory only for the sample we decode) ----
    rows, toks = [], {}
    for d_dir in ("teacher", "teacher2"):
        for f in sorted(glob.glob(str(ROOT / d_dir / "chunk_*.npz"))):
            d = np.load(f)
            off, tk = d["off"], d["tok"]
            for i in range(len(d["ids"])):
                rid, ns = str(d["ids"][i]), int(d["n_samp"][i])
                n = int(off[i + 1] - off[i])
                m = meta[rid]
                words = len(re.findall(r"\S+", TAG.sub("", m["reply"])))
                secs = (n - 1) / 8 / HZ
                rows.append(dict(id=rid, n_samp=ns, src=m["src"], group=group_of(rid), lang=m["lang"], text=m["reply"],
                                 description=m.get("description"), n_tok=n, secs=round(secs, 2),
                                 secs_per_word=round(secs / max(1, words), 3), done=bool(d["done"][i]),
                                 dir=d_dir, chunk=Path(f).name, idx=i))
    ok = [r for r in rows if r["done"]]
    print(f"samples total={len(rows)} terminated={len(ok)} dropped_unterminated={len(rows)-len(ok)}")
    by_lang = defaultdict(list)
    for r in ok:
        by_lang[r["lang"]].append(r["secs_per_word"])
    for lang, v in by_lang.items():
        v = np.array(v)
        print(f"  {lang}: secs/word p1/p10/p50/p90/p99 = {np.percentile(v,[1,10,50,90,99]).round(2).tolist()}")
    for r in ok:
        v = np.array(by_lang[r["lang"]])
        lo, hi = np.percentile(v, [1, 99])
        r["flag"] = "short" if r["secs_per_word"] < lo else "long" if r["secs_per_word"] > hi else ""
    print("  flagged outliers:", Counter(r["flag"] for r in ok))
    with open(ROOT / "manifest.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("dir", "chunk", "idx")}, ensure_ascii=False) + "\n")

    # ---- strata for listening / ASR samples ----
    def strat(n):
        buckets = defaultdict(list)
        for r in ok:
            if r["flag"]:
                continue
            tag = "tag" if TAG.search(r["text"]) else "plain"
            tone = (r["description"] or "none").split(",")[0]
            buckets[(r["group"], r["lang"], tag if tag == "tag" else tone)].append(r)
        keys = sorted(buckets)
        out, i = [], 0
        for _ in range(n):
            b = buckets[keys[i % len(keys)]]
            out.append(b.pop(rnd.randrange(len(b)))) if b else None
            i += 1
        return out

    listen = strat(args.n_wavs)
    asr = listen + [r for r in strat(args.n_asr) if r not in listen]
    asr = asr[: args.n_asr]

    # ---- 3: decode wavs ----
    mimi = MimiModel.from_pretrained(TEACHER_REPO / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(TEACHER_REPO / "codec").sampling_rate)
    cache = {}

    def get_tokens(r):
        key = (r["dir"], r["chunk"])
        if key not in cache:
            cache.clear()
            cache[key] = np.load(ROOT / r["dir"] / r["chunk"])
        d = cache[key]
        return d["tok"][d["off"][r["idx"]]: d["off"][r["idx"] + 1]]

    def to_audio(token_ids):
        values, frame, exp = [], [], 0
        for t in token_ids:
            it = mapping.get(int(t))
            if it is None:
                frame, exp = [], 0
                continue
            c, q = it
            if q == exp:
                frame.append(c); exp += 1
                if exp == 8:
                    values.extend(frame); frame, exp = [], 0
            elif q == 0:
                frame, exp = [c], 1
            else:
                frame, exp = [], 0
        with torch.inference_mode():
            codes = torch.tensor(values, device="cuda").reshape(1, -1, 8).transpose(1, 2)
            return mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()

    (ROOT / "listen").mkdir(exist_ok=True)
    audio_of = {}
    for r in sorted(asr, key=lambda r: (r["dir"], r["chunk"])):
        a = to_audio(get_tokens(r))
        audio_of[(r["id"], r["n_samp"])] = a
    idx = ["| file | lang | tone/tags | secs | text |", "|---|---|---|---|---|"]
    for r in listen:
        a = audio_of[(r["id"], r["n_samp"])]
        fn = f"{r['id']}_{r['n_samp']}.wav"
        with wave.open(str(ROOT / "listen" / fn), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((a * 32767).astype("<i2").tobytes())
        idx.append(f"| {fn} | {r['lang']} | {r['description'] or '-'} | {r['secs']} | {r['text']} |")
    (ROOT / "listen" / "index.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"wrote {len(listen)} wavs -> {ROOT/'listen'}")

    # ---- 4: ASR baseline ----
    del mimi
    torch.cuda.empty_cache()
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
    res = defaultdict(lambda: [0, 0, 0, 0])  # word_err, word_n, char_err, char_n
    per = []
    for r in asr:
        a = audio_of[(r["id"], r["n_samp"])]
        a16 = resample_poly(a, 16000, sr).astype(np.float32)
        lang = "en" if r["lang"] == "en" else "hi"
        hyp = norm_text(asr_pipe({"raw": a16, "sampling_rate": 16000}, generate_kwargs={"language": lang, "task": "transcribe"})["text"])
        ref = norm_text(r["text"])
        we, wn = edit_distance(ref.split(), hyp.split()), max(1, len(ref.split()))
        ce, cn = edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")), max(1, len(ref.replace(" ", "")))
        for key in ("all", f"lang={r['lang']}", f"group={r['group']}", f"tag={'yes' if TAG.search(r['text']) else 'no'}"):
            x = res[key]; x[0] += we; x[1] += wn; x[2] += ce; x[3] += cn
        per.append(dict(id=r["id"], n_samp=r["n_samp"], lang=r["lang"], ref=ref, hyp=hyp, wer=round(we / wn, 3), cer=round(ce / cn, 3)))
    print("\nASR baseline (whisper-large-v3-turbo; relative metric, noisy for Hinglish/numbers):")
    for k, (we, wn, ce, cn) in sorted(res.items()):
        print(f"  {k:16s} WER {100*we/wn:5.1f}%  CER {100*ce/cn:5.1f}%  (n_words={wn})")
    with open(ROOT / "asr_baseline.jsonl", "w") as f:
        for p in per:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
