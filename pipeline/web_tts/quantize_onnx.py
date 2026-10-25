"""4-bit weight-only quantisation of the exported decoder with ONNX Runtime's MatMulNBits (HQQ, block 64).
Kept in higher precision: the logits head ([2048,16385], fidelity of the sampling distribution) and layer-0 mlp.down_proj (the one real
single-matrix outlier found in quant_sensitivity). Embedding Gather, norms, attention score matmuls and the stop head are not touched."""
import argparse, time
from pathlib import Path
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="onnx_fp32/model.onnx"); ap.add_argument("--out", default="onnx_int4/model.onnx")
ap.add_argument("--bits", type=int, default=4); ap.add_argument("--block", type=int, default=64)
ap.add_argument("--algo", default="hqq", choices=["hqq", "rtn", "minmax"])
args = ap.parse_args()
t0 = time.time()
model = onnx.load(args.src, load_external_data=True)
init = {i.name: i for i in model.graph.initializer}
exclude = []
for n in model.graph.node:
    if n.op_type == "MatMul" and n.input[1] in init:
        dims = list(init[n.input[1]].dims)
        if dims == [2048, 16385] or n.input[1] == "layers.0.down":
            exclude.append(n.name)
print(f"loaded {time.time()-t0:.0f}s | excluding {len(exclude)} nodes: {exclude}", flush=True)
algo = {"hqq": lambda: HQQWeightOnlyQuantConfig(block_size=args.block, bits=args.bits), "rtn": lambda: RTNWeightOnlyQuantConfig(),
        "minmax": lambda: DefaultWeightOnlyQuantConfig(block_size=args.block, is_symmetric=False, bits=args.bits)}[args.algo]()
q = MatMulNBitsQuantizer(model, bits=args.bits, block_size=args.block, is_symmetric=False, nodes_to_exclude=exclude, algo_config=algo)
q.process()
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
q.model.save_model_to_file(str(out), use_external_data_format=True)
sz = sum(p.stat().st_size for p in out.parent.iterdir())
print(f"quantised + saved {out} in {time.time()-t0:.0f}s | total size {sz/1e9:.2f} GB", flush=True)
