"""Do the expression tags (<laugh>/<chuckle>/<sigh>) actually produce a sound?  For held-out tagged sentences generate the SAME text with and
without the tag (several seeds) and compare audio length: a rendered vocalization adds roughly 0.3-1 s. Compares the original model with the
QAT 2-bit model.  Token counts are enough (100 audio tokens = 1 s), so no codec / ASR is needed."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from quant_sweep import generate
from student_lib import Data
TEACHER = R.parent / "rumik-oss-1"
TAG = re.compile(r"<(laugh|chuckle|sigh)>")
AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")
data = Data(); val_ids = set(str(i) for i in data.ids[data.is_val])
tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
vocab = tok.get_vocab(); end_id = int(vocab["</audio>"])
allowed = torch.tensor(sorted([i for s, i in vocab.items() if AUDIO_RE.match(s)] + [end_id]))
rows = [json.loads(l) for fn in ("corpus.jsonl", "corpus2.jsonl") for l in open(R / fn)]
rng = np.random.default_rng(0)
pick = defaultdict(list)
for r in rows:
    if r["id"] in val_ids and TAG.search(r["reply"]):
        pick[("en" if r["lang"] == "en" else "hi/mixed", TAG.search(r["reply"]).group(1))].append(r)
sel = []
for k, v in sorted(pick.items()):
    rng.shuffle(v); sel += [(k, r) for r in v[:14]]
print({k: min(14, len(v)) for k, v in sorted(pick.items())}, "total", len(sel), flush=True)
def prompt(r, with_tag):
    t = r["reply"] if with_tag else re.sub(r"\s+", " ", TAG.sub(" ", r["reply"])).strip()
    body = f'<description="{r["description"]}"> {t}' if r.get("description") else t
    return tok(f"<text>Ira: {body}<audio>").input_ids
jobs = [(k, r, wt, s) for (k, r) in sel for wt in (True, False) for s in (0, 1)]
def run(model_dir, name):
    m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    order = sorted(range(len(jobs)), key=lambda i: len(prompt(jobs[i][1], jobs[i][2])))
    secs = {}
    for seed in (0, 1):
        torch.manual_seed(seed)
        idx = [i for i in order if jobs[i][3] == seed]
        for s in range(0, len(idx), 32):
            ch = idx[s:s + 32]
            gen = generate(m, [prompt(jobs[i][1], jobs[i][2]) for i in ch], allowed, end_id)
            for i, row in zip(ch, gen):
                e = np.where(row == end_id)[0]; n = int(e[0]) if len(e) else len(row)
                secs[i] = (n // 8) * 8 / 100.0
    del m; torch.cuda.empty_cache()
    res = defaultdict(list)
    for (k, r), _ in [((k, r), 0) for (k, r) in sel]:
        pass
    by = defaultdict(lambda: defaultdict(list))
    for i, (k, r, wt, s) in enumerate(jobs):
        by[(k, r["id"], s)][wt].append(secs[i])
    d = defaultdict(list)
    for (k, rid, s), v in by.items():
        d[k].append(v[True][0] - v[False][0])
    print(f"\n== {name}: extra audio seconds when the tag is present (with tag - without tag), paired by text and seed")
    for k, v in sorted(d.items()):
        v = np.array(v); print(f"  {k[0]:9s} <{k[1]}>: mean {v.mean():+.2f} s | median {np.median(v):+.2f} | share of pairs with >= +0.30 s: {100*(v>=0.3).mean():3.0f}%  (n={len(v)})")
    allv = np.concatenate([np.array(v) for v in d.values()]); print(f"  ALL: mean {allv.mean():+.2f} s | share >= +0.30 s: {100*(allv>=0.3).mean():.0f}%")
    return d
a = run(TEACHER, "original (bf16)")
b = run(R / "student/qat1/merged", "QAT 2-bit")
