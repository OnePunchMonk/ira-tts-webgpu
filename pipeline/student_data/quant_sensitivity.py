"""
Per-matrix quantisation sensitivity of the teacher + mixed-precision plans, scored TEACHER-FORCED (no sampling, no ASR):
    KD(config) = cross-entropy of the config's audio-token distribution against the saved teacher top-k (stop position excluded).
    delta = KD(config) - KD(bf16).   Deterministic, ~8 s per evaluation.

Phase 1: for every (layer, module) quantise ONLY that matrix to 2-bit HQQ g32 -> delta2.      (252 evals)
Phase 2: for every module type, quantise it in ALL layers at 2- and 3-bit -> ratio r3/r2.     (14 evals)
Phase 3: greedy mixed-precision plans (2/3/4-bit) for several average-bit budgets, evaluated for real vs uniform 2/3/4-bit.
Effective bits/weight = bits + (16 scale + bits zero)/group.   Plans are written to quant_plans/ for `quant_sweep.py --configs plan:<file>`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from quant_sweep import LIN, apply_plan, hqq_deq  # noqa: E402
from student_lib import Data, distill_loss  # noqa: E402

TEACHER = R.parent / "rumik-oss-1"
OUT_ROWS = torch.tensor([277394] + list(range(261008, 261008 + 16384)))
OPTS = {2: 32, 3: 64, 4: 64}  # bits -> group size used in plans


def eff_bits(b, g):
    return b + (16 + b) / g


def build_batches(data, tok, n, seed=1):
    text_of = {}
    for fn in ("corpus.jsonl", "corpus2.jsonl"):
        for l in open(R / fn):
            d = json.loads(l)
            text_of[d["id"]] = f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"]
    rng = np.random.default_rng(seed)
    val = rng.choice(data.val_idx, n, replace=False)
    orig = data.collate

    def collate(g, r=None, a=0.0):
        b = orig(g, r, a)
        rows = []
        for i in g:
            p = tok("<text>Ira: " + text_of[str(data.ids[i])] + "<audio>").input_ids
            o = data.otok[data.ooff[i]:data.ooff[i + 1]]
            rows.append((np.concatenate([p, np.where(o[:-1] == 0, 277394, o[:-1] - 1 + 261008)]), len(p)))
        L = max(len(x[0]) for x in rows)
        ids = torch.zeros(len(g), L, dtype=torch.long); attn = torch.zeros(len(g), L, dtype=torch.long); pos = []
        for k, (x, P) in enumerate(rows):
            ids[k, :len(x)] = torch.from_numpy(x); attn[k, :len(x)] = 1
            pos.append(torch.arange(P - 1, P - 1 + data.ooff[g[k] + 1] - data.ooff[g[k]]))
        b["ids"], b["attn"], b["pos"] = ids, attn, torch.cat(pos)
        return b

    data.collate = collate
    return list(data.batches(val, 24000, rng, shuffle=False))


@torch.no_grad()
def kd_of(model, batches):
    kd = n = 0
    for b in batches:
        _, k, _ = distill_loss(model, b, OUT_ROWS, stop_w=0.0, ce_w=0.0)
        nb = int((b["tgt"] != 0).sum()); kd += float(k) * nb; n += nb
    return kd / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--budgets", default="2.8,3.0,3.3,3.6")
    ap.add_argument("--skip-phase1", action="store_true")
    args = ap.parse_args()
    data = Data()
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    batches = build_batches(data, tok, args.n_samples)
    model = AutoModelForCausalLM.from_pretrained(TEACHER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    orig = {(li, n): layer.get_submodule(n).weight.data.detach().cpu().clone()
            for li, layer in enumerate(model.model.layers) for n in LIN}
    nL = len(model.model.layers)

    def restore(keys):
        with torch.no_grad():
            for li, n in keys:
                model.model.layers[li].get_submodule(n).weight.data.copy_(orig[(li, n)].cuda())

    def eval_plan(plan):
        apply_plan(model, orig, plan)
        v = kd_of(model, batches)
        restore([(int(k.split("|")[0]), k.split("|", 1)[1]) for k in plan])
        return v

    t0 = time.time()
    base = kd_of(model, batches)
    print(f"bf16 KD {base:.4f}  ({sum(int((b['tgt']!=0).sum()) for b in batches)} tokens)", flush=True)
    (R / "quant_plans").mkdir(exist_ok=True)
    sens_path = R / "quant_plans" / "sensitivity.json"

    # ---------- phase 1 ----------
    if args.skip_phase1 and sens_path.exists():
        d2 = {k: v for k, v in json.load(open(sens_path))["d2"].items()}
    else:
        d2 = {}
        for li in range(nL):
            for n in LIN:
                d2[f"{li}|{n}"] = eval_plan({f"{li}|{n}": [2, OPTS[2]]}) - base
            print(f"layer {li:2d} | " + " ".join(f"{n.split('.')[1][:4]}={d2[f'{li}|{n}']:+.3f}" for n in LIN)
                  + f" | sum {sum(d2[f'{li}|{n}'] for n in LIN):+.3f} | {time.time()-t0:.0f}s", flush=True)
        json.dump(dict(base=base, d2=d2), open(sens_path, "w"))

    # ---------- phase 2: per-type ratios ----------
    ratio = {}
    for n in LIN:
        a = eval_plan({f"{li}|{n}": [2, OPTS[2]] for li in range(nL)}) - base
        b3 = eval_plan({f"{li}|{n}": [3, OPTS[3]] for li in range(nL)}) - base
        ratio[n] = float(np.clip(b3 / a, 0.0, 1.0)) if a > 1e-4 else 0.0
        print(f"type {n:18s} all-layers 2-bit {a:+.3f} | 3-bit {b3:+.3f} | r3/r2 {ratio[n]:.2f}", flush=True)

    # ---------- phase 3: greedy plans ----------
    size = {n: orig[(0, n)].numel() for n in LIN}
    keys = [(li, n) for li in range(nL) for n in LIN]
    total = sum(size[n] for _, n in keys)
    def delta(li, n, b):  # estimated delta at bit-width b
        d = max(d2[f"{li}|{n}"], 0.0)
        return d if b == 2 else d * ratio[n] if b == 3 else 0.0
    refs = {}
    for b in (2, 3, 4):
        refs[f"uniform{b}b"] = (eval_plan({f"{li}|{n}": [b, OPTS[b]] for li, n in keys}) - base, sum(eff_bits(b, OPTS[b]) * size[n] for _, n in keys) / total)
        print(f"uniform {b}-bit g{OPTS[b]} | eff {refs[f'uniform{b}b'][1]:.2f} bits | delta {refs[f'uniform{b}b'][0]:+.3f}", flush=True)
    results = {}
    for B in map(float, args.budgets.split(",")):
        cur = {k: 2 for k in keys}
        bits_now = sum(eff_bits(2, OPTS[2]) * size[n] for _, n in keys)
        while True:
            best = None
            for (li, n), b in cur.items():
                if b == 4: continue
                nb = b + 1
                dcost = (eff_bits(nb, OPTS[nb]) - eff_bits(b, OPTS[b])) * size[n]
                if (bits_now + dcost) / total > B: continue
                gain = (delta(li, n, b) - delta(li, n, nb)) / dcost
                if best is None or gain > best[0]: best = (gain, li, n, nb, dcost)
            if best is None or best[0] <= 0: break
            _, li, n, nb, dc = best; cur[(li, n)] = nb; bits_now += dc
        plan = {f"{li}|{n}": [b, OPTS[b]] for (li, n), b in cur.items()}
        est = sum(delta(li, n, b) for (li, n), b in cur.items())
        real = eval_plan(plan) - base
        mix = {b: sum(size[n] for (li, n), x in cur.items() if x == b) / total for b in (2, 3, 4)}
        json.dump(plan, open(R / "quant_plans" / f"plan_{B}.json", "w"))
        results[B] = dict(eff_bits=bits_now / total, est=est, real=real, mix=mix)
        print(f"plan budget {B}: eff {bits_now/total:.2f} bits | est delta {est:+.3f} | REAL delta {real:+.3f} | "
              f"share of params at 2/3/4-bit {mix[2]:.2f}/{mix[3]:.2f}/{mix[4]:.2f}", flush=True)
    json.dump(dict(base=base, d2=d2, ratio=ratio, refs=refs, plans=results), open(sens_path, "w"))
    top = sorted(d2.items(), key=lambda kv: -kv[1])[:12]
    print("\nmost fragile matrices at 2-bit:", [(k, round(v, 3)) for k, v in top])
    print("DONE", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
