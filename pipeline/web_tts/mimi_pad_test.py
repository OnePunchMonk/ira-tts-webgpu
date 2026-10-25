"""Can the 32-codebook Mimi ONNX decoder (onnx-community/kyutai-mimi-ONNX) decode Ira's 8-codebook codes?
Pad codebooks 8..31 with the minimum-norm code of each codebook and compare waveforms with the HF 8-codebook decode."""
import json, sys
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from transformers import AutoTokenizer, MimiModel
ROOT = Path(__file__).resolve().parent.parent
m = MimiModel.from_pretrained(ROOT / "rumik-oss-1" / "codec", dtype=torch.float32).eval()
q = m.quantizer
embeds = [q.semantic_residual_vector_quantizer.layers[0].codebook.embed] + [l.codebook.embed for l in q.acoustic_residual_vector_quantizer.layers]
norms = torch.stack([e.norm(dim=-1) for e in embeds])                       # [32, 2048]
null = norms.argmin(-1)                                                     # min-norm code per codebook
print("min embedding norm per codebook (first 12):", [round(float(norms[k].min()), 3) for k in range(12)], "| mean norm cb0 %.2f cb8 %.2f cb31 %.2f" % (norms[0].mean(), norms[8].mean(), norms[31].mean()))
json.dump([int(x) for x in null], open("mimi_null_codes.json", "w"))
tok = AutoTokenizer.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True)
vocab = tok.get_vocab(); import re
mp = {i: (int(a.group(1)), int(a.group(2))) for s, i in vocab.items() if (a := re.match(r"^<(\d+)_(\d+)>$", s))}
d = np.load(ROOT / "student_data" / "teacher" / "chunk_00000.npz"); off = d["off"]
res = []
for si in (3, 40, 90):
    toks = d["tok"][off[si]: off[si + 1] - 1]
    vals = [mp[int(t)][0] for t in toks]; nfr = len(vals) // 8
    codes = torch.tensor(vals[: nfr * 8]).reshape(nfr, 8).T[None]               # [1,8,T]
    with torch.no_grad():
        ref = m.decode(codes).audio_values[0, 0].numpy()
    pad = torch.cat([codes, null[8:, None].expand(24, nfr)[None]], 1)           # [1,32,T]
    for f in ("decoder_model.onnx", "decoder_model_fp16.onnx"):
        s = ort.InferenceSession(f"mimi_onnx/onnx/{f}", providers=["CPUExecutionProvider"])
        out = s.run(None, {"audio_codes": pad.numpy().astype(np.int64)})[0][0, 0]
        n = min(len(ref), len(out)); a, b = ref[:n], out[:n]
        snr = 10 * np.log10((a ** 2).sum() / (((a - b) ** 2).sum() + 1e-12))
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"sample {si} {f:24s} frames {nfr} | len ref {len(ref)} onnx {len(out)} | SNR {snr:5.1f} dB | corr {corr:.4f}")
    # sanity: how much would ZERO-codes (code 0) cost instead of min-norm codes
    pad0 = torch.cat([codes, torch.zeros(1, 24, nfr, dtype=torch.long)], 1)
    out0 = ort.InferenceSession("mimi_onnx/onnx/decoder_model.onnx", providers=["CPUExecutionProvider"]).run(None, {"audio_codes": pad0.numpy()})[0][0, 0][:n]
    print(f"           (code-0 padding instead: SNR {10*np.log10((a**2).sum()/(((a-out0)**2).sum()+1e-12)):.1f} dB)")
