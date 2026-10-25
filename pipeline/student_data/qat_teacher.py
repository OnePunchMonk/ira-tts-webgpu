"""
QAT of the TEACHER's decoder weights to ~2.56 effective bits (2-bit, group 32) with KD against the saved teacher logprobs.

 * every Linear in the 36 decoder layers -> QLinear: fp32 latent weight + learnable per-group scale/zero (zero rounded with STE),
   fake-quant in the forward pass (round STE, LSQ-style gradients to scale). Init: per-group MSE-optimal clip search.
 * layer-0 mlp.down_proj kept at 4-bit g64 (the one real outlier in quant_sensitivity).
 * embeddings / norms / stop_predictor frozen. Loss = top-k KD on audio positions (stop position excluded: the teacher's stop head,
   which stays frozen, makes that decision).  Inputs = exact teacher prompt ("Ira: ...").
 * exports merged fake-quant bf16 weights (HF-loadable dir) so quant_sweep.py --model-dir can score real audio.

    python qat_teacher.py --out student/qat1 --epochs 1
Never run alongside another GPU job; keep the clock cap on.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from quant_sensitivity import OUT_ROWS, build_batches, kd_of  # noqa: E402
from quant_sweep import LIN  # noqa: E402
from student_lib import Data, distill_loss  # noqa: E402

TEACHER = R.parent / "rumik-oss-1"


def ste_round(x):
    return (x.round() - x).detach() + x


@torch.no_grad()
def init_scale_zero(W, bits, g):
    o, i = W.shape
    x = W.float().view(o, i // g, g)
    qmax = 2 ** bits - 1
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    c, half = (mn + mx) / 2, (mx - mn) / 2
    best = torch.full_like(mn, float("inf")); bs = torch.ones_like(mn); bz = torch.zeros_like(mn)
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
        lo, hi = c - half * r, c + half * r
        s = ((hi - lo) / qmax).clamp(min=1e-8)
        z = torch.round(-lo / s)
        err = ((torch.clamp(torch.round(x / s) + z, 0, qmax) - z) * s - x).pow(2).sum(-1, keepdim=True)
        upd = err < best
        best = torch.where(upd, err, best); bs = torch.where(upd, s, bs); bz = torch.where(upd, z, bz)
    return bs, bz


class QLinear(nn.Module):
    def __init__(self, lin, bits, g):
        super().__init__()
        self.bits, self.g, self.qmax = bits, g, 2 ** bits - 1
        W = lin.weight.data.float()
        self.weight = nn.Parameter(W.clone())
        s, z = init_scale_zero(W, bits, g)
        self.scale, self.zero = nn.Parameter(s), nn.Parameter(z)
        self.out_features, self.in_features = W.shape

    def w_hat(self, dtype):
        o, i = self.weight.shape
        x = self.weight.view(o, i // self.g, self.g)
        s = self.scale.abs().clamp(min=1e-8)
        z = ste_round(self.zero)
        q = ste_round(torch.clamp(x / s + z, 0, self.qmax))
        return ((q - z) * s).view(o, i).to(dtype)

    def forward(self, x):
        return F.linear(x, self.w_hat(x.dtype))


def eff_bits(b, g):
    return b + (16 + b) / g


@torch.no_grad()
def export_merged(model, out_dir, src=TEACHER):
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QLinear):
            sd[name + ".weight"] = mod.w_hat(torch.bfloat16).cpu().contiguous()
    for k, v in model.state_dict().items():
        mod_name = k.rsplit(".", 1)[0]
        if k.endswith((".scale", ".zero")) or (k.endswith(".weight") and k in sd):
            continue
        sd[k] = v.detach().to(torch.bfloat16).cpu().contiguous() if v.dtype.is_floating_point else v.cpu()
    sd.pop("lm_head.weight", None)  # tied to embed_tokens
    save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "configuration_tinyaya.py", "modeling_tinyaya.py", "generation_config.json"):
        shutil.copy(src / f, out_dir / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--l0-down-bits", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--micro-tokens", type=int, default=8000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--export-every", type=int, default=75)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    data = Data()
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    val_batches = build_batches(data, tok, 16)  # also installs the teacher-prompt collate on `data`
    model = AutoModelForCausalLM.from_pretrained(TEACHER, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    model.eval()
    base = kd_of(model, val_batches)
    print(f"bf16 teacher val KD {base:.4f}", flush=True)

    n_q = tot_bits = tot_par = 0
    for li, layer in enumerate(model.model.layers):
        for name in LIN:
            parent = layer.get_submodule(name.rsplit(".", 1)[0]); leaf = name.rsplit(".", 1)[1]
            lin = getattr(parent, leaf)
            b, g = (args.l0_down_bits, 64) if (li == 0 and name == "mlp.down_proj") else (args.bits, args.group)
            setattr(parent, leaf, QLinear(lin, b, g).cuda())
            n_q += 1; tot_par += lin.weight.numel(); tot_bits += eff_bits(b, g) * lin.weight.numel()
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for m in model.modules():
        if isinstance(m, QLinear):
            for p in m.parameters():
                p.requires_grad = True; trainable.append(p)
    print(f"quantised {n_q} matrices | {tot_par/1e9:.2f}B params | avg {tot_bits/tot_par:.2f} effective bits | "
          f"trainable {sum(p.numel() for p in trainable)/1e9:.2f}B", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    kd0 = kd_of(model, val_batches)
    print(f"step 0 | PTQ-init val KD {kd0:.4f} (delta vs bf16 {kd0-base:+.3f})", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    rng = np.random.default_rng(0)
    n_micro = sum(1 for _ in data.batches(data.train_idx, args.micro_tokens, np.random.default_rng(1), shuffle=False))
    steps_per_epoch = n_micro // args.accum
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"micro-batches/epoch {n_micro} | steps/epoch {steps_per_epoch} | total steps {total}", flush=True)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    model.train()
    step, micro, t0, best, hist = 0, 0, time.time(), float("inf"), []
    acc = 0.0
    done = False
    while not done:
        for b in data.batches(data.train_idx, args.micro_tokens, rng):
            loss, kd, _ = distill_loss(model, b, OUT_ROWS, stop_w=0.0, ce_w=0.0)
            (loss / args.accum).backward()
            acc += float(kd); micro += 1
            if micro % args.accum:
                continue
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            el = time.time() - t0
            print(f"step {step}/{total} | train KD {acc/args.accum:.3f} | gn {float(gn):.2f} lr {lr_at(step):.1e} | "
                  f"{el/step:.0f}s/step | {el/60:.0f} min | ETA {(total-step)*el/step/60:.0f} min", flush=True)
            acc = 0.0
            if step % args.eval_every == 0 or step >= total:
                model.eval(); kd_v = kd_of(model, val_batches); model.train()
                hist.append(dict(step=step, val_kd=kd_v, delta=kd_v - base))
                json.dump(dict(base=base, ptq_init=kd0, hist=hist), open(out / "history.json", "w"))
                tag = ""
                if kd_v < best:
                    best = kd_v; tag = " *best*"
                print(f"== step {step} | val KD {kd_v:.4f} (delta vs bf16 {kd_v-base:+.3f}){tag}", flush=True)
            if step % args.export_every == 0 or step >= total:
                model.eval(); export_merged(model, out / "merged"); model.train()
                print(f"exported merged fake-quant weights -> {out/'merged'}", flush=True)
            if step >= total:
                done = True
                break
    print(f"done. best val KD {best:.4f} (bf16 {base:.4f}, PTQ-init {kd0:.4f})", flush=True)


if __name__ == "__main__":
    main()
