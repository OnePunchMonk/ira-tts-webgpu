"""
Distil the teacher into the pruned student: KL on the teacher's saved top-K logprobs (+ small CE on the sampled token,
</audio> position up-weighted). fp32 master weights, bf16 autocast, AdamW, cosine LR.

    python train_student.py --init student/init_12L_4096 --out student/run1 --epochs 8
Never run while another GPU job / inference is active (shared GB10); keep the clock cap on.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Cohere2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_lib import Data, distill_loss  # noqa: E402


@torch.no_grad()
def evaluate(model, data, idx, max_tokens):
    model.eval()
    kd = ce = n = 0
    for b in data.batches(idx, max_tokens, np.random.default_rng(0), shuffle=False):
        _, k, c = distill_loss(model, b, data.out_rows)
        nb = b["tgt"].numel(); kd += float(k) * nb; ce += float(c) * nb; n += nb
    model.train()
    return kd / n, ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--micro-tokens", type=int, default=16000, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--aug-prob", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--sched-steps", type=int, default=None, help="length of the LR schedule (default: total); lets a short run follow a longer run's LR curve")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    data = Data()
    model = Cohere2ForCausalLM.from_pretrained(args.init, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    model.train()

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)

    rng = np.random.default_rng(0)
    # steps per epoch estimated from one pass of batch counts
    n_micro = sum(1 for _ in data.batches(data.train_idx, args.micro_tokens, np.random.default_rng(1), shuffle=False))
    steps_per_epoch = n_micro // args.accum
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"train samples {len(data.train_idx)} | micro-batches/epoch {n_micro} | steps/epoch {steps_per_epoch} | total steps {total}", flush=True)
    val_idx = np.random.default_rng(2).choice(data.val_idx, min(args.eval_n, len(data.val_idx)), replace=False)

    sched = args.sched_steps or total

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, sched - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    step, best, t0, tok_seen = 0, float("inf"), time.time(), 0
    hist = []
    kd, ce = evaluate(model, data, val_idx, args.micro_tokens)
    print(f"step 0 | val KD {kd:.3f} CE {ce:.3f}", flush=True)
    done = False
    while not done:
        micro = 0
        acc_l = acc_k = acc_c = 0.0
        for b in data.batches(data.train_idx, args.micro_tokens, rng, aug_prob=args.aug_prob):
            loss, k, c = distill_loss(model, b, data.out_rows)
            (loss / args.accum).backward()
            acc_l += float(loss); acc_k += float(k); acc_c += float(c); tok_seen += int(b["tgt"].numel()); micro += 1
            if micro % args.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} | loss {acc_l/args.accum:.3f} KD {acc_k/args.accum:.3f} CE {acc_c/args.accum:.3f} | "
                      f"gn {float(gn):.2f} lr {lr_at(step):.2e} | {tok_seen/el:.0f} tok/s | {el/60:.1f} min", flush=True)
            acc_l = acc_k = acc_c = 0.0
            if step % args.eval_every == 0 or step >= total:
                kd, ce = evaluate(model, data, val_idx, args.micro_tokens)
                hist.append(dict(step=step, val_kd=kd, val_ce=ce))
                tag = ""
                if kd < best:
                    best = kd; model.save_pretrained(out / "best", safe_serialization=True); tag = " *best*"
                print(f"== step {step} | val KD {kd:.3f} CE {ce:.3f}{tag}", flush=True)
                json.dump(hist, open(out / "history.json", "w"))
            if step >= total:
                done = True
                break
    model.save_pretrained(out / "final", safe_serialization=True)
    print(f"done. best val KD {best:.3f}; final -> {out/'final'}", flush=True)


if __name__ == "__main__":
    main()
