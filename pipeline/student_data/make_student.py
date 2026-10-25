"""
Initialise the student from the teacher: pruned vocabulary, selected layers, importance-pruned MLPs.

    python make_student.py --layers 12 --mlp 4096 --out student/init_12L_4096
    python make_student.py --layers 36 --mlp 11008 --check      # unpruned control: must reproduce teacher logprobs

Layer choice: evenly spaced teacher layers (always incl. first and last) with their native attention types.
MLP pruning: keep the top-M intermediate neurons per layer by mean |silu(gate)*up| on calibration data
(teacher run on real audio-token sequences), slicing gate/up rows and down columns together.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Cohere2Config, Cohere2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_lib import ROOT, Data, distill_loss  # noqa: E402

TEACHER = Path(__file__).resolve().parent.parent / "rumik-oss-1"
T_AUDIO0 = 261008
SPECIAL_TEACHER = {0: 0, 1: 2, 2: 277392, 3: 277393, 4: 277394, 5: 277401, 6: 277402, 7: 277403}  # student row -> teacher id


def student_to_teacher_ids(ids, vocab):
    """student embed-row ids -> teacher token ids (for calibration / check forward through the teacher)."""
    tt = np.array(vocab["teacher_text_ids"])
    out = np.zeros_like(ids)
    for r, t in SPECIAL_TEACHER.items():
        out[ids == r] = t
    m = (ids >= vocab["text_base"]) & (ids < vocab["text_base"] + len(tt))
    out[m] = tt[ids[m] - vocab["text_base"]]
    m = ids >= vocab["audio_base"]
    out[m] = ids[m] - vocab["audio_base"] + T_AUDIO0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--mlp", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", type=int, default=96, help="calibration samples for MLP importance")
    ap.add_argument("--check", action="store_true", help="compare student vs saved teacher logprobs on val samples")
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / f"init_{args.layers}L_{args.mlp}"
    data = Data()
    V = data.vocab

    os.environ.setdefault("HF_MODULES_CACHE", str(Path(__file__).resolve().parent.parent / ".hf_modules_cache"))
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    tcfg = teacher.config
    n_t = tcfg.num_hidden_layers
    picks = sorted({int(round(x)) for x in np.linspace(0, n_t - 1, args.layers)})
    assert len(picks) == args.layers, "layer picks collided"
    print("teacher layers kept:", picks)

    # ---- MLP importance from calibration data ----
    M, full_M = args.mlp, tcfg.intermediate_size
    keep_idx = {}
    if M < full_M:
        rng = np.random.default_rng(0)
        cal = rng.choice(data.train_idx, args.calib, replace=False)
        scores = {i: torch.zeros(full_M, device="cuda") for i in picks}
        ntok = 0
        hooks = []
        for i in picks:
            def hook(mod, inp, i=i):
                scores[i] += inp[0].float().abs().sum(dim=(0, 1))
            hooks.append(teacher.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook))
        with torch.inference_mode():
            for i in cal:
                ids, P, *_ = data.sample(int(i))
                t_ids = torch.tensor(student_to_teacher_ids(ids, V))[None].cuda()
                teacher.model(input_ids=t_ids)
                ntok += t_ids.shape[1]
        for h in hooks:
            h.remove()
        for i in picks:
            keep_idx[i] = torch.topk(scores[i] / ntok, M).indices.sort().values
        print(f"MLP importance from {ntok} tokens; keeping {M}/{full_M} neurons per layer")

    # ---- build student ----
    cfg = Cohere2Config(
        vocab_size=V["n_vocab"], hidden_size=tcfg.hidden_size, intermediate_size=M, num_hidden_layers=args.layers,
        num_attention_heads=tcfg.num_attention_heads, num_key_value_heads=tcfg.num_key_value_heads, head_dim=tcfg.head_dim,
        hidden_act=tcfg.hidden_act, max_position_embeddings=tcfg.max_position_embeddings, layer_norm_eps=tcfg.layer_norm_eps,
        rope_theta=tcfg.rope_theta, sliding_window=tcfg.sliding_window, logit_scale=tcfg.logit_scale,
        layer_types=[tcfg.layer_types[i] for i in picks], attention_bias=False, attention_dropout=0.0,
        tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=4, use_cache=True,
    )
    student = Cohere2ForCausalLM(cfg).to(torch.float32)
    sd_t = teacher.model.state_dict()

    with torch.no_grad():
        E = sd_t["embed_tokens.weight"].float()
        W = student.model.embed_tokens.weight
        W.zero_()
        for r, t in SPECIAL_TEACHER.items():
            W[r] = E[t]
        W[8:16] = E[:].mean(0)
        tb = V["text_base"]
        W[tb: tb + len(V["teacher_text_ids"])] = E[torch.tensor(V["teacher_text_ids"])]
        W[V["audio_base"]: V["audio_base"] + 16384] = E[T_AUDIO0: T_AUDIO0 + 16384]
        student.model.norm.weight.copy_(sd_t["norm.weight"].float())
        for j, i in enumerate(picks):
            layer = student.model.layers[j]
            for name, p in layer.named_parameters():
                src = sd_t[f"layers.{i}.{name}"].float()
                if M < full_M and ".mlp." in f".{name}":
                    k = keep_idx[i].to(src.device)
                    if name.startswith("mlp.down_proj"):
                        src = src[:, k]
                    else:
                        src = src[k]
                p.copy_(src)
    n_par = sum(p.numel() for p in student.parameters())
    print(f"student params: {n_par/1e6:.1f}M  (embeddings {student.model.embed_tokens.weight.numel()/1e6:.1f}M)")
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    json.dump(dict(layers=args.layers, mlp=M, picks=picks, params_M=round(n_par / 1e6, 1)), open(out / "student_meta.json", "w"))
    print("saved ->", out)

    # ---- evaluation on val: loss vs teacher top-k (and, for the unpruned control, a hard correctness check) ----
    student.cuda().eval()
    rng = np.random.default_rng(1)
    val = rng.choice(data.val_idx, 24, replace=False)
    tot = kd = ce = ref = n = 0
    for batch in data.batches(val, 24000, rng, shuffle=False):
        with torch.no_grad():
            l, k, c = distill_loss(student, batch, data.out_rows)
        valid = batch["tkid"] >= 0
        pt = torch.softmax(batch["tklp"].masked_fill(~valid, -1e4), -1) * valid
        wt = torch.where(batch["tgt"] == 0, torch.full_like(batch["tklp"][:, 0], 20.0), torch.ones_like(batch["tklp"][:, 0]))
        r = (-(pt * batch["tklp"].masked_fill(~valid, 0)).sum(-1) * wt).sum() / wt.sum()  # teacher scoring itself
        nb = batch["tgt"].numel()
        tot, kd, ce, ref, n = tot + float(l) * nb, kd + float(k) * nb, ce + float(c) * nb, ref + float(r) * nb, n + nb
    print(f"val (24 samples, {n} tokens): weighted loss {tot/n:.3f} | KD xent {kd/n:.3f} (teacher scoring itself: {ref/n:.3f}) | CE(sampled) {ce/n:.3f}")
    if args.check:
        print(f"-> unpruned control: KD xent should be ~equal to the teacher's self-score; gap = {kd/n - ref/n:+.3f}")


if __name__ == "__main__":
    main()
