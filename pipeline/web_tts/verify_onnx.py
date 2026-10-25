"""Run an exported ONNX decoder (CPU EP) on the same real sequence as verify_parity.py and compare with the PyTorch reference (fp32)."""
import sys, time
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_decoder import ROOT, T_AUDIO0, build_from_teacher, empty_past
from transformers import AutoTokenizer
path = sys.argv[1]
dec, cfg, V = build_from_teacher(torch.float32)
tok = AutoTokenizer.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True)
t2s = {t: V["text_base"] + j for j, t in enumerate(V["teacher_text_ids"])}
special = {2: 1, 277392: 2, 277393: 3, 277394: 4, 277401: 5, 277402: 6, 277403: 7}
to_p = lambda i: special[i] if i in special else (V["audio_base"] + i - T_AUDIO0 if T_AUDIO0 <= i < T_AUDIO0 + 16384 else t2s[i])
d = np.load(ROOT / "student_data" / "teacher" / "chunk_00000.npz"); off = d["off"]
ids = tok("<text>Ira: " + str(d["prompt_txt"][3]) + "<audio>").input_ids + d["tok"][off[3]: off[3] + 60].tolist()
pr = torch.tensor([[to_p(i) for i in ids]]); pos = torch.arange(len(ids))[None]
so = ort.SessionOptions(); so.intra_op_num_threads = 8
t0 = time.time(); sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); print(f"session {time.time()-t0:.0f}s", flush=True)
n = cfg["num_hidden_layers"]
KV = np.float16 if "float16" in sess.get_inputs()[2].type else np.float32
print("kv dtype", KV.__name__, flush=True)
feed = {"input_ids": pr.numpy(), "position_ids": pos.numpy()}
for i in range(n):
    for kv in ("key", "value"):
        feed[f"past_key_values.{i}.{kv}"] = np.zeros((1, cfg["num_key_value_heads"], 0, cfg["head_dim"]), KV)
t0 = time.time(); out = sess.run(None, feed); print(f"prefill {len(ids)} tokens: {time.time()-t0:.1f}s", flush=True)
with torch.no_grad():
    ref, rstop, *_ = dec(pr, pos, *empty_past(cfg))
a, b = torch.tensor(out[0][0]), ref[0]
print(f"ONNX vs PyTorch fp32 (last position): max|diff| {float((a-b).abs().max()):.3f} | cosine {float(torch.cosine_similarity(a,b,dim=0)):.5f} | "
      f"argmax equal {int(a.argmax())==int(b.argmax())} | top5 overlap {len(set(a.topk(5).indices.tolist())&set(b.topk(5).indices.tolist()))}/5 | "
      f"KL(ref||onnx) {float(torch.nn.functional.kl_div(torch.log_softmax(a,-1), torch.softmax(b,-1), reduction='sum')):.4f}")
# one cached decode step through ONNX
past = out[2:]
feed2 = {"input_ids": pr[:, -1:].numpy(), "position_ids": pos[:, -1:].numpy() + 1}
for i in range(n):
    feed2[f"past_key_values.{i}.key"], feed2[f"past_key_values.{i}.value"] = past[2*i], past[2*i+1]
t0 = time.time(); o2 = sess.run(None, feed2); print(f"one cached decode step: {time.time()-t0:.2f}s | logits shape {o2[0].shape} | present kv len {o2[2].shape[2]}", flush=True)
