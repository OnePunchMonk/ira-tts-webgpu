"""
How low can we quantise the TEACHER before its speech breaks?

Fake-quantises (quantise -> dequantise back to bf16) every Linear in the 36 decoder layers (q,k,v,o,gate,up,down) with
RTN or HQQ at a given bit-width / group size, generates the held-out texts with a batched sampler that reproduces the
teacher's decode (audio-only logits, temp 0.8, top-k 30, stop_predictor head), decodes with Mimi and scores with
Whisper-large-v3-turbo. Embeddings (tied lm_head), norms and the stop head stay bf16 (embeddings would be 8-bit in a real export).

    python quant_sweep.py --configs bf16,rtn4g32,hqq4g64,hqq3g32,hqq2g16 --per-lang 15 --tag sweep1
Config syntax:  bf16 | rtn{bits}g{group} | hqq{bits}g{group}
Run only when no other GPU job is active.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel, pipeline

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from check_teacher import TAG, edit_distance, norm_text  # noqa: E402

TEACHER = R.parent / "rumik-oss-1"
LIN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")


def rtn(w, bits, g):
    """asymmetric min-max group-wise RTN along the input dim."""
    o, i = w.shape
    x = w.float().reshape(o, i // g, g)
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8)
    zp = torch.round(-mn / scale)
    q = (torch.round(x / scale) + zp).clamp(0, 2 ** bits - 1)
    return ((q - zp) * scale).reshape(o, i).to(w.dtype)


def hqq_deq(w, bits, g):
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    lin.weight.data.copy_(w)
    q = HQQLinear(lin, BaseQuantizeConfig(nbits=bits, group_size=g, axis=1), compute_dtype=w.dtype, device=str(w.device))
    return q.dequantize().to(w.dtype)


def apply_plan(model, orig, plan):
    """plan: {"<layer>|<module>": [bits, group]} -> mixed-precision HQQ fake-quant (unlisted matrices stay bf16)."""
    with torch.no_grad():
        for key, (bits, g) in plan.items():
            li, name = key.split("|", 1)
            li = int(li)
            model.model.layers[li].get_submodule(name).weight.data.copy_(hqq_deq(orig[(li, name)].cuda(), int(bits), int(g)))


def apply_quant(model, orig, cfg):
    if cfg == "bf16":
        return
    if cfg.startswith("plan:"):
        return apply_plan(model, orig, json.load(open(cfg[5:])))
    m = re.fullmatch(r"(rtn|hqq)(\d)g(\d+)", cfg)
    kind, bits, g = m.group(1), int(m.group(2)), int(m.group(3))
    fn = rtn if kind == "rtn" else hqq_deq
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            for name in LIN:
                mod = layer.get_submodule(name)
                mod.weight.data.copy_(fn(orig[(li, name)].cuda(), bits, g))


@torch.inference_mode()
def generate(model, prompts, allowed, end_id, max_new=1500, temp=0.8, topk=30):
    B = len(prompts)
    L = max(map(len, prompts))
    ids = torch.zeros(B, L, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.long)
    for k, p in enumerate(prompts):
        ids[k, L - len(p):] = torch.tensor(p); attn[k, L - len(p):] = 1
    ids, attn = ids.cuda(), attn.cuda()
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    cache, logits, hid = out.past_key_values, out.logits[:, -1], out.hidden_states[-1][:, -1]
    cur = attn.sum(-1)
    done = torch.zeros(B, dtype=torch.bool, device="cuda")
    allowed = allowed.cuda()
    gen = []
    for step in range(max_new):
        masked = torch.full_like(logits, float("-inf"))
        masked[:, allowed] = logits[:, allowed]
        if step < 8:
            masked[:, end_id] = float("-inf")
        s = masked.float() / temp
        kth = torch.topk(s, topk, dim=-1).values[:, -1:]
        s = s.masked_fill(s < kth, float("-inf"))
        tok = torch.multinomial(torch.softmax(s, -1), 1).squeeze(1)
        if step >= 8:
            stop = torch.sigmoid(model.stop_predictor(hid).squeeze(-1).float()) > 0.5
            tok = torch.where(stop, torch.full_like(tok, end_id), tok)
        tok = torch.where(done, torch.full_like(tok, end_id), tok)
        done = done | (tok == end_id)
        gen.append(tok)
        if bool(done.all()):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device="cuda")], 1)
        out = model(input_ids=tok[:, None], attention_mask=attn, position_ids=cur[:, None], past_key_values=cache,
                    use_cache=True, output_hidden_states=True)
        cache, logits, hid = out.past_key_values, out.logits[:, -1], out.hidden_states[-1][:, -1]
        cur = cur + 1
    return torch.stack(gen, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="bf16,rtn4g32,hqq4g64,hqq3g32,hqq2g16")
    ap.add_argument("--per-lang", type=int, default=15)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--langs", default=None, help="comma list to restrict languages, e.g. en")
    ap.add_argument("--save-wavs", type=int, default=0, help="save the first N decoded utterances per config to listen_quant/<tag>_<cfg>/")
    ap.add_argument("--model-dir", default=None, help="load weights from this HF dir instead of the teacher (e.g. QAT export)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    vocab = tok.get_vocab()
    end_id = int(vocab["</audio>"])
    audio_ids = sorted(i for s, i in vocab.items() if AUDIO_RE.match(s))
    allowed = torch.tensor(sorted(audio_ids + [end_id]))
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}

    text_of = {}
    for fn in ("corpus.jsonl", "corpus2.jsonl"):
        for l in open(R / fn):
            d = json.loads(l)
            text_of[d["id"]] = (d, f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"])
    base = [json.loads(l) for l in open(R / "asr_baseline.jsonl")]
    by_lang, seen = defaultdict(list), set()
    for b in base:
        if b["id"] in seen: continue
        seen.add(b["id"]); by_lang[b["lang"]].append(b["id"])
    rng = np.random.default_rng(0)
    items = []
    for lang, ids_ in sorted(by_lang.items()):
        if args.langs and lang not in args.langs.split(","):
            continue
        for i in rng.choice(ids_, min(args.per_lang, len(ids_)), replace=False):
            d, txt = text_of[str(i)]
            items.append(dict(id=str(i), lang=lang, text=d["reply"], prompt=tok(f"<text>Ira: {txt}<audio>").input_ids))
    items.sort(key=lambda x: len(x["prompt"]))
    print(f"{len(items)} held-out texts: " + str({k: min(args.per_lang, len(v)) for k, v in by_lang.items()}), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir or TEACHER, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    orig = {(li, n): layer.get_submodule(n).weight.data.detach().cpu().clone()
            for li, layer in enumerate(model.model.layers) for n in LIN}
    mimi = MimiModel.from_pretrained(TEACHER / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(TEACHER / "codec").sampling_rate)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")

    results = {}
    for cfg in args.configs.split(","):
        t0 = time.time()
        # restore originals then quantise
        with torch.no_grad():
            for (li, n), w in orig.items():
                model.model.layers[li].get_submodule(n).weight.data.copy_(w.cuda())
        apply_quant(model, orig, cfg)
        t_q = time.time() - t0
        torch.manual_seed(args.seed)
        toks = {}
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            gen = generate(model, [x["prompt"] for x in chunk], allowed, end_id)
            for x, row in zip(chunk, gen):
                e = np.where(row == end_id)[0]
                toks[x["id"]] = (row[: e[0]] if len(e) else row, len(e) > 0)
        res = defaultdict(lambda: dict(w=[], c=[], spw=[], term=[], wc=[]))
        per_item = []
        for x in items:
            row, term = toks[x["id"]]
            a = np.array([t for t in row if int(t) in mapping])
            nfr = len(a) // 8
            vals = torch.tensor([[mapping[int(t)][0] for t in a[: nfr * 8]]]).reshape(nfr, 8).cuda() if nfr else None
            if vals is None:
                audio = np.zeros(sr, np.float32)
            else:
                with torch.inference_mode():
                    audio = mimi.decode(vals.T[None]).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
            if args.save_wavs and sum(1 for _ in (R / "listen_quant" / f"{args.tag}_{cfg.replace(':', '_').replace('/', '_')[-24:]}").glob("*.wav")) < args.save_wavs:
                import wave
                wd = R / "listen_quant" / f"{args.tag}_{cfg.replace(':', '_').replace('/', '_')[-24:]}"; wd.mkdir(parents=True, exist_ok=True)
                with wave.open(str(wd / f"{x['lang']}_{x['id']}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((audio * 32767).astype("<i2").tobytes())
            lang = "en" if x["lang"] == "en" else "hi"
            hyp = norm_text(asr({"raw": resample_poly(audio, 16000, sr).astype(np.float32), "sampling_rate": 16000},
                                generate_kwargs={"language": lang, "task": "transcribe"})["text"])
            ref = norm_text(x["text"])
            we, wn = edit_distance(ref.split(), hyp.split()), max(1, len(ref.split()))
            ce, cn = edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")), max(1, len(ref.replace(" ", "")))
            per_item.append(dict(id=x["id"], lang=x["lang"], ref=ref, hyp=hyp, wer=we / wn, cer=ce / cn, secs=len(audio) / sr, terminated=bool(term), n_tok=int(len(row))))
            for key in ("all", x["lang"]):
                r = res[key]; r["w"].append(we / wn); r["c"].append(ce / cn); r["wc"].append(min(1.0, we / wn))
                r["spw"].append(len(audio) / sr / max(1, len(re.findall(r"\S+", TAG.sub("", x["text"]))))); r["term"].append(term)
        summ = {k: dict(n=len(v["w"]), wer=float(np.mean(v["w"])), wercap=float(np.mean(v["wc"])), wermed=float(np.median(v["w"])), loops=int(sum(1 for w_ in v["w"] if w_ > 2.0)), cer=float(np.mean(v["c"])), spw=float(np.median(v["spw"])),
                        term=float(np.mean(v["term"]))) for k, v in res.items()}
        results[cfg] = summ
        json.dump(per_item, open(R / f"quant_{args.tag}_items.json", "w"), ensure_ascii=False)
        line = " | ".join(f"{k}: WER {100*v['wer']:.1f} capped {100*v['wercap']:.1f} med {100*v['wermed']:.1f} loops {v['loops']}" for k, v in sorted(summ.items()))
        print(f"[{cfg:8s}] quant {t_q:.0f}s total {time.time()-t0:.0f}s | stop-rate {summ['all']['term']:.2f} s/word {summ['all']['spw']:.2f} | {line}", flush=True)
        json.dump(results, open(R / f"quant_{args.tag}.json", "w"), indent=1)
    print("\n== summary (mean per-utterance WER/CER %, same texts and seed; bf16 = control)")
    for cfg, summ in results.items():
        print(f"{cfg:9s} " + "  ".join(f"{k}={100*v['wer']:.0f}/{100*v['cer']:.0f}" for k, v in sorted(summ.items())))


if __name__ == "__main__":
    main()
