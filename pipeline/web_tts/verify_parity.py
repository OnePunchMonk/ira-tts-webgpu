"""Parity: ref_decoder.WebDecoder (pruned vocab, explicit KV) vs the Hugging Face teacher, on a real prompt + generated audio tokens (CPU)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_decoder import ROOT, T_AUDIO0, VOCAB, build_from_teacher, empty_past
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(8)
t0 = time.time()
dec, cfg, V = build_from_teacher(torch.float32)
print(f"web decoder built {time.time()-t0:.0f}s", flush=True)
tok = AutoTokenizer.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True)
hf = AutoModelForCausalLM.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True, dtype=torch.float32).eval()
# map teacher ids -> pruned ids
t2s = {t: V["text_base"] + j for j, t in enumerate(V["teacher_text_ids"])}
special = {2: 1, 277392: 2, 277393: 3, 277394: 4, 277401: 5, 277402: 6, 277403: 7}
def to_pruned(i):
    if i in special: return special[i]
    if i >= T_AUDIO0 and i < T_AUDIO0 + 16384: return V["audio_base"] + i - T_AUDIO0
    if i in t2s: return t2s[i]
    raise KeyError(i)
d = np.load(ROOT / "student_data" / "teacher" / "chunk_00000.npz")
txt = str(d["prompt_txt"][3]); off = d["off"]; gen = d["tok"][off[3]: off[3] + 60].tolist()
ids = tok("<text>Ira: " + txt + "<audio>").input_ids + gen
print("text:", txt[:80], "| tokens", len(ids), flush=True)
rows = [277394] + list(range(T_AUDIO0, T_AUDIO0 + 16384))
with torch.no_grad():
    ref_all = hf(input_ids=torch.tensor([ids])).logits[0]            # [T, 277404]
    ref = ref_all[:, rows]                                            # restricted to audio + </audio>
    pr = torch.tensor([[to_pruned(i) for i in ids]])
    pos = torch.arange(len(ids))[None]
    lg, stop, *pres = dec(pr, pos, *empty_past(cfg))                  # full prefill, last position
    a = lg[0]; b = ref[-1]
    print(f"prefill last-pos: max|diff| {float((a-b).abs().max()):.4f} | cosine {float(torch.cosine_similarity(a,b,dim=0)):.6f} | "
          f"argmax equal {int(a.argmax())==int(b.argmax())} | top5 overlap {len(set(a.topk(5).indices.tolist())&set(b.topk(5).indices.tolist()))}/5", flush=True)
    # KV-cache path: prefill all but the last token, then decode the last token
    lg0, _, *p0 = dec(pr[:, :-1], pos[:, :-1], *empty_past(cfg))
    lg1, stop1, *_ = dec(pr[:, -1:], pos[:, -1:], *p0)
    print(f"cached decode vs full prefill: max|diff| {float((lg1-lg).abs().max()):.5f} | stop prob {float(stop1):.4f} vs {float(stop):.4f}", flush=True)
    # teacher's own stop head on HF hidden state
    hs = hf(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][:, -1:]
    print(f"HF stop prob {float(torch.sigmoid(hf.stop_predictor(hs)).squeeze()):.4f}")
    # a few more positions through the cached path (prompt length ~ 100): compare logits at positions of generated tokens
    n_ok = 0
    P0 = len(ids) - 8
    lgx, _, *pk = dec(pr[:, :P0], pos[:, :P0], *empty_past(cfg))
    for j in range(P0, len(ids)):
        lgj, _, *pk = dec(pr[:, j:j+1], pos[:, j:j+1], *pk)
        n_ok += int(lgj[0].argmax()) == int(ref[j].argmax())
        print(f"  pos {j}: max|diff| {float((lgj[0]-ref[j]).abs().max()):.4f}", end="")
    print(f"\nargmax agreement over 8 cached decode steps: {n_ok}/8")
