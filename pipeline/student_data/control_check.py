"""Control: unpruned student (pruned vocab only) scored on the EXACT teacher prompt ('Ira: ' prefix kept, tokenised
the way the teacher saw it), stop position excluded. KD xent should match the teacher's self-score closely."""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, Cohere2ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_lib import Data, distill_loss
R = Path(__file__).resolve().parent
ctrl = sys.argv[1]
data = Data(); V = data.vocab
tok = AutoTokenizer.from_pretrained(R.parent / "rumik-oss-1", trust_remote_code=True)
t2s = {t: V["text_base"] + j for j, t in enumerate(V["teacher_text_ids"])}
special = {int(k): v for k, v in V["teacher_special_ids"].items()}
inv = {i: s for s, i in tok.get_vocab().items()}
u2b = {}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
text_of = {}
for fn in ("corpus.jsonl", "corpus2.jsonl"):
    for l in open(R / fn):
        d = json.loads(l); text_of[d["id"]] = f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"]
def prompt_ids(txt):
    ids = [1]
    for t in tok("<text>Ira: " + txt + "<audio>", add_special_tokens=False).input_ids:
        if t in special: ids.append(special[t])
        elif t in t2s: ids.append(t2s[t])
        else: ids.extend(V["byte_sid"][str(u2b[ch])] for ch in inv[t])
    return np.array(ids, np.int64)
orig = data.sample
def sample(i, rng=None, aug_prob=0.0):
    ids, P, o, ti, tl = orig(i)
    newp = prompt_ids(text_of[str(data.ids[i])])
    return np.concatenate([newp, ids[P:]]), len(newp), o, ti, tl
data.sample = sample
student = Cohere2ForCausalLM.from_pretrained(ctrl, dtype=torch.float32).cuda().eval()
rng = np.random.default_rng(1); val = rng.choice(data.val_idx, 24, replace=False)
tot = kd = ref = n = 0
for batch in data.batches(val, 24000, rng, shuffle=False):
    with torch.no_grad(): l, k, c = distill_loss(student, batch, data.out_rows, stop_w=0.0)
    valid = batch["tkid"] >= 0
    pt = torch.softmax(batch["tklp"].masked_fill(~valid, -1e4), -1) * valid
    m = (batch["tgt"] != 0).float()
    r = (-(pt * batch["tklp"].masked_fill(~valid, 0)).sum(-1) * m).sum() / m.sum()
    nb = batch["tgt"].numel(); kd += float(k) * nb; ref += float(r) * nb; n += nb
print(f"exact-prompt, stop excluded: student KD xent {kd/n:.3f} vs teacher self-score {ref/n:.3f}  (gap {kd/n-ref/n:+.3f})")
