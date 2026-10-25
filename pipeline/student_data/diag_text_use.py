"""Does the model use the text? Score val KD with the correct prompt vs a prompt from a DIFFERENT sample (audio targets unchanged)."""
import sys
from pathlib import Path
import numpy as np, torch
from transformers import Cohere2ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_lib import Data, distill_loss
data = Data(); ck = sys.argv[1]
model = Cohere2ForCausalLM.from_pretrained(ck, dtype=torch.float32).cuda().eval()
rng = np.random.default_rng(3); val = rng.choice(data.val_idx, 96, replace=False)
orig = data.sample
def run(shuffle):
    perm = np.roll(val, 1)
    mp = {int(a): int(b) for a, b in zip(val, perm)}
    def sample(i, r=None, aug_prob=0.0):
        ids, P, o, ti, tl = orig(i)
        if shuffle:
            ids2, P2, *_ = orig(mp[int(i)])
            ids = np.concatenate([ids2[:P2], ids[P:]]); P = P2
        return ids, P, o, ti, tl
    data.sample = sample
    kd = n = 0; early = ne = 0
    for b in data.batches(val, 16000, np.random.default_rng(0), shuffle=False):
        with torch.no_grad(): _, k, c = distill_loss(model, b, data.out_rows)
        nb = b["tgt"].numel(); kd += float(k) * nb; n += nb
    return kd / n
a, s = run(False), run(True)
print(f"student KD correct text {a:.3f} | shuffled text {s:.3f} | text benefit {s-a:+.3f} nats/token")
