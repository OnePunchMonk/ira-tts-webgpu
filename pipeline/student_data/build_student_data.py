"""
Build the student's pruned vocabulary and training arrays from the teacher data.

Student id layout (Cohere2 embedding rows, tied input/output):
    0 pad | 1 BOS | 2 <text> | 3 <audio> | 4 </audio> | 5 <laugh> | 6 <chuckle> | 7 <sigh> | 8..15 reserved
    TEXT_BASE=16 .. TEXT_BASE+n_text-1     text tokens kept from the teacher tokenizer (incl. all 256 byte tokens)
    AUDIO_BASE .. AUDIO_BASE+16383          Mimi tokens, a = value*8 + quantizer  (teacher id = 261008 + a)
Output ("out") index space for logits/targets: 0 = </audio>, 1+a = audio token a  (16385 classes, matches the
teacher's restricted distribution). Prompt = BOS <text> {description?} text <audio>; speaker prefix dropped (single voice).

Teacher tokens outside the kept set are decomposed into byte tokens (browser tokenizer does the same), and
train_student.py additionally decomposes a few percent of random tokens so the student learns byte-level text too.

Outputs (student_data/student/):
    vocab.json   {"teacher_ids":[...student id order for ids>=16...], "specials":..., "audio_base":..., "decomp":{sid:[sids]}}
    data.npz     per-sample ragged arrays + split flags
"""
import argparse
import glob
import json
import re
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT.parent / "rumik-oss-1"
AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")
T_AUDIO0 = 261008
SPECIAL_T = {"<text>": 2, "<audio>": 3, "</audio>": 4, "<laugh>": 5, "<chuckle>": 6, "<sigh>": 7}
TEXT_BASE = 16


def bytes_to_unicode():  # GPT-2 byte<->char table used by byte-level BPE
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-text", type=int, default=16000, help="text tokens to keep (incl. 256 byte tokens)")
    ap.add_argument("--min-count", type=int, default=3, help="corpus count needed to be force-kept")
    ap.add_argument("--extra-convs", type=int, default=15000, help="samvaad convs for broad frequency stats")
    ap.add_argument("--n-val-extra", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = ROOT / "student"
    out.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)  # default regex = what tokenizer.json ships (same as browser)
    vocab = tok.get_vocab()
    inv = {i: s for s, i in vocab.items()}
    special_t = {tid: SPECIAL_T[s] for s, tid in ((s, vocab[s]) for s in SPECIAL_T)}
    b2u = bytes_to_unicode()
    byte_tids = [vocab[b2u[b]] for b in range(256)]  # all 256 byte tokens must exist
    print("byte tokens found:", len(byte_tids))

    # ---- samples (clean, non-outlier) ----
    manifest = {(m["id"], m["n_samp"]): m for m in map(json.loads, open(ROOT / "manifest.jsonl")) if not m["flag"]}
    print("clean samples:", len(manifest))
    samples = []
    for d_dir in ("teacher", "teacher2"):
        for f in sorted(glob.glob(str(ROOT / d_dir / "chunk_*.npz"))):
            d = np.load(f)
            # NpzFile re-reads the whole array on EVERY d[...] access, and a slice keeps its base array alive,
            # so load each array exactly once per chunk and .copy() the slices (else RAM blows up per-sample).
            off, ids_, ns_, txt_ = d["off"], d["ids"], d["n_samp"], d["prompt_txt"]
            tok_, tki_, tkl_ = d["tok"], d["topk_id"], d["topk_lp"]
            for i in range(len(ids_)):
                key = (str(ids_[i]), int(ns_[i]))
                if key not in manifest:
                    continue
                s, e = off[i], off[i + 1]
                samples.append(dict(id=key[0], n_samp=key[1], text=str(txt_[i]), tok=tok_[s:e].copy(),
                                    tk_id=tki_[s:e].copy(), tk_lp=tkl_[s:e].copy()))
            del tok_, tki_, tkl_
    print("loaded samples:", len(samples))

    # ---- text vocabulary: corpus tokens + broad Hindi/English stats ----
    def enc(txt):
        return tok(txt, add_special_tokens=False).input_ids
    cnt = Counter()
    ptxt = {s["id"]: s["text"] for s in samples}
    for t in ptxt.values():
        cnt.update(enc(t))
    corpus_cnt = Counter(cnt)
    sv = pd.read_parquet(ROOT / "raw/samvaad/data/train-00000-of-00001.parquet")
    for msgs in sv.messages[: args.extra_convs]:
        for m in msgs:
            cnt.update(enc(m["content"][:1500]))
    audio_set = {i for s, i in vocab.items() if AUDIO_RE.match(s)}
    banned = audio_set | set(special_t) | {2, 3, 6, 277393, 277394}
    keep = list(dict.fromkeys(byte_tids))
    forced = [t for t, c in corpus_cnt.most_common() if c >= args.min_count and t not in banned and t not in set(keep)]
    keep += forced
    for t, c in cnt.most_common():
        if len(keep) >= args.n_text:
            break
        if t not in banned and t not in set(keep):
            keep.append(t)
    keep = keep[: max(args.n_text, len(byte_tids) + len(forced))]
    print(f"text tokens kept: {len(keep)} (bytes 256, corpus-forced {len(forced)})")
    t2s = {t: TEXT_BASE + j for j, t in enumerate(keep)}
    audio_base = TEXT_BASE + len(keep)
    audio_base += (-audio_base) % 8
    n_vocab = audio_base + 16384

    # decomposition of every kept token to byte tokens (for OOV + augmentation)
    u2b = {u: b for b, u in b2u.items()}
    byte_sid = {b: t2s[byte_tids[b]] for b in range(256)}

    def decomp_token(tid):
        s = inv[tid]
        return [byte_sid[u2b[ch]] for ch in s if ch in u2b]

    decomp = {str(t2s[t]): decomp_token(t) for t in keep if len(inv[t]) > 1}

    def to_student_prompt(txt):
        ids = [1, 2]  # BOS, <text>
        for t in enc(txt):
            if t in special_t:
                ids.append(special_t[t])
            elif t in t2s:
                ids.append(t2s[t])
            else:  # OOV -> bytes
                ids.extend(decomp_token(t))
        ids.append(3)  # <audio>
        return ids

    # ---- split by text ----
    asr_ids = {json.loads(l)["id"] for l in open(ROOT / "asr_baseline.jsonl")}
    all_ids = sorted({s["id"] for s in samples})
    rest = [i for i in all_ids if i not in asr_ids]
    val_ids = set(asr_ids) | set(rnd.sample(rest, args.n_val_extra))
    print(f"val texts: {len(val_ids)} (asr baseline {len(asr_ids)})")

    # ---- convert ----
    ptok, poff, otok, ooff, tk_id, tk_lp, is_val, sid_list, n_samp = [], [0], [], [0], [], [], [], [], []
    oov_tokens = tot_tokens = 0
    for s in samples:
        p = to_student_prompt(s["text"])
        ptok.extend(p); poff.append(len(ptok))
        o = np.where(s["tok"] == 277394, 0, s["tok"] - T_AUDIO0 + 1).astype(np.int32)
        assert o.min() >= 0 and o.max() <= 16384, "unexpected non-audio teacher token"
        otok.append(o); ooff.append(ooff[-1] + len(o))
        ti = s["tk_id"]
        # entries with logprob < -50 are padding: at the stop position the teacher forces </audio> to p=1 and the
        # other 19 slots hold arbitrary ids at -inf; short top-k rows are padded with id -1 / lp -1e4
        invalid = (ti < 0) | (s["tk_lp"].astype(np.float32) < -50)
        oi = np.where(ti == 277394, 0, ti - T_AUDIO0 + 1).astype(np.int32)
        oi[invalid] = -1
        s["tk_lp"] = np.where(invalid, np.float16(-1e4), s["tk_lp"]).astype(np.float16)
        assert oi.max() <= 16384 and ((oi >= 0) | invalid).all(), "top-k contained a real non-audio token"
        tk_id.append(oi); tk_lp.append(s["tk_lp"])
        is_val.append(s["id"] in val_ids); sid_list.append(s["id"]); n_samp.append(s["n_samp"])
        raw = enc(s["text"]); tot_tokens += len(raw); oov_tokens += sum(t not in t2s and t not in special_t for t in raw)
    print(f"prompt OOV (decomposed to bytes): {oov_tokens}/{tot_tokens} = {100*oov_tokens/tot_tokens:.2f}% of text tokens")

    np.savez(out / "data.npz", ptok=np.array(ptok, np.int32), poff=np.array(poff, np.int64), otok=np.concatenate(otok),
             ooff=np.array(ooff, np.int64), tk_id=np.concatenate(tk_id), tk_lp=np.concatenate(tk_lp),
             is_val=np.array(is_val), ids=np.array(sid_list), n_samp=np.array(n_samp, np.int16))
    json.dump(dict(teacher_text_ids=keep, text_base=TEXT_BASE, audio_base=audio_base, n_vocab=n_vocab,
                   specials={"pad": 0, "bos": 1, "text": 2, "audio": 3, "end": 4, "laugh": 5, "chuckle": 6, "sigh": 7},
                   teacher_special_ids={str(t): s for t, s in special_t.items()}, decomp=decomp,
                   byte_sid={str(b): s for b, s in byte_sid.items()}),
              open(out / "vocab.json", "w"))
    n_tr = sum(not v for v in is_val)
    print(f"student vocab {n_vocab} (text {len(keep)}, audio 16384) | train samples {n_tr}, val samples {len(is_val)-n_tr}")
    print(f"out tokens: train {sum(len(o) for o, v in zip(otok, is_val) if not v)}, total {ooff[-1]}")


if __name__ == "__main__":
    main()
