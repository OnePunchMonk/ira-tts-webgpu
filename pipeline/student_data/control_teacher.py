"""Score the ORIGINAL bf16 teacher (full vocab, HF) on the exact teacher prompts vs the saved NVFP4/vLLM top-k.
Isolates the bf16-vs-NVFP4 gap from any student vocab-mapping error."""
import json, sys, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_lib import Data, distill_loss
R = Path(__file__).resolve().parent
data = Data()
tok = AutoTokenizer.from_pretrained(R.parent / "rumik-oss-1", trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(R.parent / "rumik-oss-1", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
text_of = {}
for fn in ("corpus.jsonl", "corpus2.jsonl"):
    for l in open(R / fn):
        d = json.loads(l); text_of[d["id"]] = f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"]
orig = data.collate
def collate(g, rng=None, aug_prob=0.0):
    b = orig(g, rng, aug_prob)
    # rebuild input ids in TEACHER ids: exact teacher prompt + audio inputs from the stored tokens
    rows = []
    for i in g:
        p = tok("<text>Ira: " + text_of[str(data.ids[i])] + "<audio>").input_ids
        o = data.otok[data.ooff[i]:data.ooff[i + 1]]
        a = np.where(o[:-1] == 0, 277394, o[:-1] - 1 + 261008)
        rows.append((np.concatenate([p, a]), len(p)))
    L = max(len(r[0]) for r in rows)
    ids = torch.zeros(len(g), L, dtype=torch.long); attn = torch.zeros(len(g), L, dtype=torch.long)
    pos = []
    for k, (x, P) in enumerate(rows):
        ids[k, :len(x)] = torch.from_numpy(x); attn[k, :len(x)] = 1
        T = data.ooff[g[k] + 1] - data.ooff[g[k]]; pos.append(torch.arange(P - 1, P - 1 + T))
    b["ids"], b["attn"], b["pos"] = ids, attn, torch.cat(pos)
    return b
data.collate = collate
out_rows = torch.tensor([277394] + list(range(261008, 261008 + 16384)))
rng = np.random.default_rng(1); val = rng.choice(data.val_idx, 24, replace=False)
kd = ref = n = 0
for batch in data.batches(val, 24000, rng, shuffle=False):
    with torch.no_grad(): l, k, c = distill_loss(teacher, batch, out_rows, stop_w=0.0)
    valid = batch["tkid"] >= 0
    pt = torch.softmax(batch["tklp"].masked_fill(~valid, -1e4), -1) * valid
    m = (batch["tgt"] != 0).float()
    r = (-(pt * batch["tklp"].masked_fill(~valid, 0)).sum(-1) * m).sum() / m.sum()
    nb = batch["tgt"].numel(); kd += float(k) * nb; ref += float(r) * nb; n += nb
print(f"ORIGINAL bf16 teacher vs saved NVFP4 top-k, stop excluded: KD xent {kd/n:.3f} vs self-score {ref/n:.3f} (gap {kd/n-ref/n:+.3f})")
