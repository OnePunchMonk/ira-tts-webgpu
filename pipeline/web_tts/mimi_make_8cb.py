"""Zero the embedding tables of Mimi codebooks 8..31 inside the 32-codebook ONNX decoder, so feeding any code (we feed 0) for them
contributes exactly nothing -> bit-for-bit the same maths as the HF 8-codebook decode that Ira's codes require."""
import json, re
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import numpy_helper
from transformers import MimiModel
ROOT = Path(__file__).resolve().parent.parent
src = "mimi_onnx/onnx/decoder_model.onnx"; dst = "mimi_onnx/decoder_8cb_fp32.onnx"
m = onnx.load(src)
init = {i.name: i for i in m.graph.initializer}
gathers = {n.name: n for n in m.graph.node if n.op_type == "Gather" and n.name.startswith("/Gather") and init.get(n.input[0]) is not None and list(init[n.input[0]].dims) == [2048, 256]}
order = sorted(gathers, key=lambda s: int(re.sub(r"\D", "", s) or 0))
print("codebook gathers found:", len(order), "first/last:", order[0], order[-1])
assert len(order) == 32
for k in range(8, 32):
    t = init[gathers[order[k]].input[0]]
    t.CopyFrom(numpy_helper.from_array(np.zeros((2048, 256), np.float32), t.name))
onnx.save(m, dst)
print("saved", dst, round(Path(dst).stat().st_size / 1e6), "MB")
# verify against HF 8-codebook decode on real teacher codes
hf = MimiModel.from_pretrained(ROOT / "rumik-oss-1" / "codec", dtype=torch.float32).eval()
tokz = None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True)
mp = {i: (int(a.group(1)), int(a.group(2))) for s, i in tok.get_vocab().items() if (a := re.match(r"^<(\d+)_(\d+)>$", s))}
d = np.load(ROOT / "student_data" / "teacher" / "chunk_00000.npz"); off = d["off"]
sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
for si in (3, 40, 90, 150):
    toks = d["tok"][off[si]: off[si + 1] - 1]
    vals = [mp[int(t)][0] for t in toks]; nfr = len(vals) // 8
    codes = torch.tensor(vals[: nfr * 8]).reshape(nfr, 8).T[None]
    with torch.no_grad(): ref = hf.decode(codes).audio_values[0, 0].numpy()
    pad = torch.cat([codes, torch.zeros(1, 24, nfr, dtype=torch.long)], 1).numpy()
    out = sess.run(None, {"audio_codes": pad})[0][0, 0]
    n = min(len(ref), len(out))
    snr = 10 * np.log10((ref[:n] ** 2).sum() / (((ref[:n] - out[:n]) ** 2).sum() + 1e-20))
    print(f"sample {si}: frames {nfr} | len ref {len(ref)} onnx {len(out)} | SNR {snr:.1f} dB | max|diff| {np.abs(ref[:n]-out[:n]).max():.2e}")
