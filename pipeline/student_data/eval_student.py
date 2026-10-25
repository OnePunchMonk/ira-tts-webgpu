"""
Generate audio from the student on the held-out ASR-baseline texts, decode with Mimi, and score with the same
Whisper ASR as the teacher baseline (asr_baseline.jsonl) so WER/CER are directly comparable.

    python eval_student.py --ckpt student/run1/best --tag run1

Also writes listen_student/<tag>/*.wav (student) next to the teacher wavs in listen/ where the text matches.
Sampling matches the teacher data: temperature 0.8, top-k 30, logits restricted to audio + </audio>.
"""
import argparse
import json
import re
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import (AutoFeatureExtractor, AutoTokenizer, Cohere2ForCausalLM, LogitsProcessor, LogitsProcessorList,
                          MimiModel, pipeline)

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from check_teacher import TAG, edit_distance, norm_text  # noqa: E402
from student_lib import Data  # noqa: E402

TEACHER = R.parent / "rumik-oss-1"


class AudioOnly(LogitsProcessor):
    def __init__(self, rows, n_vocab):
        self.mask = torch.full((n_vocab,), float("-inf"))
        self.mask[rows] = 0.0

    def __call__(self, input_ids, scores):
        return scores + self.mask.to(scores.device, scores.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=2100)
    ap.add_argument("--n-wavs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    data = Data(); V = data.vocab

    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    tv = tok.get_vocab(); inv = {i: s for s, i in tv.items()}
    t2s = {t: V["text_base"] + j for j, t in enumerate(V["teacher_text_ids"])}
    special = {int(k): v for k, v in V["teacher_special_ids"].items()}
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}

    def prompt_ids(txt):  # student prompt: BOS <text> text <audio> (no speaker prefix)
        ids = [1]
        for t in tok("<text>" + txt + "<audio>", add_special_tokens=False).input_ids:
            if t in special: ids.append(special[t])
            elif t in t2s: ids.append(t2s[t])
            else: ids.extend(V["byte_sid"][str(u2b[ch])] for ch in inv[t])
        return ids

    # texts = the ASR baseline set (held out), in the same order, so the teacher baseline lines up
    base = [json.loads(l) for l in open(R / "asr_baseline.jsonl")]
    text_of = {}
    for fn in ("corpus.jsonl", "corpus2.jsonl"):
        for l in open(R / fn):
            d = json.loads(l); text_of[d["id"]] = (d, f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"])
    items = []
    seen = set()
    for b in base:
        if b["id"] in seen: continue
        seen.add(b["id"]); d, txt = text_of[b["id"]]
        items.append(dict(id=b["id"], lang=d["lang"], text=d["reply"], prompt_txt=txt, teacher_wer=b["wer"], teacher_cer=b["cer"]))
    print(f"{len(items)} held-out texts")

    model = Cohere2ForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).cuda().eval()
    lp = LogitsProcessorList([AudioOnly(data.out_rows, V["n_vocab"])])
    tok_out = {}
    t0 = time.time(); total_tok = 0
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt_txt"]))
    for s in range(0, len(order), args.batch):
        g = order[s:s + args.batch]
        P = [prompt_ids(items[i]["prompt_txt"]) for i in g]
        L = max(map(len, P))
        ids = torch.zeros(len(g), L, dtype=torch.long); attn = torch.zeros(len(g), L, dtype=torch.long)
        for k, p in enumerate(P):  # left pad
            ids[k, L - len(p):] = torch.tensor(p); attn[k, L - len(p):] = 1
        with torch.inference_mode():
            out = model.generate(input_ids=ids.cuda(), attention_mask=attn.cuda(), do_sample=True, temperature=0.8, top_k=30,
                                 max_new_tokens=args.max_new, eos_token_id=4, pad_token_id=0, logits_processor=lp)
        gen = out[:, L:].cpu().numpy()
        for k, i in enumerate(g):
            row = gen[k]; e = np.where(row == 4)[0]
            terminated = len(e) > 0
            row = row[: e[0]] if terminated else row
            tok_out[i] = (row, terminated); total_tok += len(row)
        print(f"  generated {min(s + args.batch, len(order))}/{len(order)} | {total_tok / (time.time() - t0):.0f} tok/s", flush=True)

    # ---- decode ----
    mimi = MimiModel.from_pretrained(TEACHER / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(TEACHER / "codec").sampling_rate)
    audio = {}
    for i, (row, term) in tok_out.items():
        a = row - V["audio_base"]  # student audio ids -> a = value*8+q
        a = a[(a >= 0) & (a < 16384)]
        nfr = len(a) // 8
        if nfr == 0:
            audio[i] = np.zeros(sr, np.float32); continue
        # frame/quantizer consistency: token j must have q = j % 8
        a = a[: nfr * 8].reshape(nfr, 8)
        codes = torch.tensor(a // 8).T[None].cuda()  # [1,8,T]: value of each quantizer
        # rows are (value*8+q); recover value per position, quantizer must equal column index
        vals = torch.tensor(a // 8).cuda()
        with torch.inference_mode():
            audio[i] = mimi.decode(vals.T[None]).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
    q_ok = np.mean([((r[0] - V["audio_base"]) % 8 == np.arange(len(r[0])) % 8).mean() if len(r[0]) else 1 for r in tok_out.values()])
    term_rate = np.mean([t for _, t in tok_out.values()])
    print(f"quantizer-order consistency {q_ok:.4f} | terminated {term_rate:.3f}")

    # ---- wavs ----
    wd = R / "listen_student" / args.tag; wd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(items), min(args.n_wavs, len(items)), replace=False):
        with wave.open(str(wd / f"{items[i]['id']}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((audio[i] * 32767).astype("<i2").tobytes())

    # ---- ASR ----
    del mimi, model; torch.cuda.empty_cache()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
    res = defaultdict(lambda: [0, 0, 0, 0]); tres = defaultdict(lambda: [0, 0, 0, 0]); spw = defaultdict(list)
    per = []
    for i, it in enumerate(items):
        a16 = resample_poly(audio[i], 16000, sr).astype(np.float32)
        lang = "en" if it["lang"] == "en" else "hi"
        hyp = norm_text(asr({"raw": a16, "sampling_rate": 16000}, generate_kwargs={"language": lang, "task": "transcribe"})["text"])
        ref = norm_text(it["text"])
        we, wn = edit_distance(ref.split(), hyp.split()), max(1, len(ref.split()))
        ce, cn = edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")), max(1, len(ref.replace(" ", "")))
        for key in ("all", f"lang={it['lang']}"):
            x = res[key]; x[0] += we; x[1] += wn; x[2] += ce; x[3] += cn
        spw[it["lang"]].append(len(audio[i]) / sr / max(1, len(re.findall(r"\S+", TAG.sub("", it["text"])))))
        per.append(dict(id=it["id"], lang=it["lang"], ref=ref, hyp=hyp, wer=round(we / wn, 3), cer=round(ce / cn, 3),
                        teacher_wer=it["teacher_wer"], teacher_cer=it["teacher_cer"]))
    # teacher baseline on the SAME texts (one sample each, as stored)
    tb = {}
    for key in ("all", "en", "hi", "mixed", "roman"):
        rows = [p for p in per if key == "all" or p["lang"] == key]
        tb[key] = (np.mean([p["teacher_wer"] for p in rows]), np.mean([p["teacher_cer"] for p in rows]))
    print(f"\n== {args.tag}: student vs teacher (mean per-utterance WER / CER, same held-out texts, Whisper large-v3-turbo)")
    for key in ("all", "lang=en", "lang=hi", "lang=mixed", "lang=roman"):
        if key not in res: continue
        k = key.split("=")[-1]; rows = [p for p in per if key == "all" or p["lang"] == k]
        sw, sc = np.mean([p["wer"] for p in rows]), np.mean([p["cer"] for p in rows])
        print(f"  {key:12s} n={len(rows):3d}  student WER {100*sw:5.1f}% CER {100*sc:5.1f}%   | teacher WER {100*tb[k][0]:5.1f}% CER {100*tb[k][1]:5.1f}%")
    print("  secs/word (student):", {k: round(float(np.median(v)), 2) for k, v in spw.items()}, "(teacher ~0.45)")
    json.dump(per, open(R / f"eval_{args.tag}.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
