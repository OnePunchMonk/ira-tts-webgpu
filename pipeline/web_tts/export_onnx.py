"""Export the pruned-vocab teacher decoder (ref_decoder.WebDecoder) to ONNX with an explicit KV cache (fp32 graph, external data)."""
import argparse, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_decoder import build_from_teacher, empty_past

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="onnx_fp32/model.onnx")
ap.add_argument("--weights-from", default=None, help="HF dir with (QAT-merged) weights instead of the teacher")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"], help="activation/weight dtype of the exported graph (fp32 islands for LN/softmax are kept)")
args = ap.parse_args()
t0 = time.time()
DT = torch.float16 if args.dtype == 'fp16' else torch.float32
dec, cfg, V = build_from_teacher(DT)
n = cfg["num_hidden_layers"]
print(f"built {time.time()-t0:.0f}s, params {sum(p.numel() for p in dec.parameters())/1e9:.2f}B", flush=True)
T, P = 3, 5
ids = torch.randint(16, 1000, (1, T)); pos = torch.arange(P, P + T)[None]
past = [(torch.randn(1, cfg["num_key_value_heads"], P, cfg["head_dim"]) * 0.1).to(DT) for _ in range(2 * n)]
in_names = ["input_ids", "position_ids"] + [f"past_key_values.{i}.{kv}" for i in range(n) for kv in ("key", "value")]
out_names = ["logits", "stop"] + [f"present.{i}.{kv}" for i in range(n) for kv in ("key", "value")]
dyn = {"input_ids": {1: "T"}, "position_ids": {1: "T"}}
for i in range(n):
    for kv in ("key", "value"):
        dyn[f"past_key_values.{i}.{kv}"] = {2: "P"}; dyn[f"present.{i}.{kv}"] = {2: "PT"}
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(dec, (ids, pos, *past), str(out), input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)
print(f"exported -> {out} in {time.time()-t0:.0f}s; files:", [(p.name, round(p.stat().st_size/1e9, 2)) for p in sorted(out.parent.iterdir())][:6], flush=True)
