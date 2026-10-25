"""
Build the decoder ONNX directly with ONNX Runtime contrib ops (the way onnxruntime-genai does) instead of tracing PyTorch:

  per layer:  LayerNormalization(fp32 residual) -> Cast fp16 -> MatMul(fused QKV) -> GroupQueryAttention(packed QKV, rotary inside, KV cache)
              -> MatMul(O) ; MatMul(fused gate+up) -> Split -> SiLU*up -> MatMul(down) ; residual += both (fp32)
  ~12 nodes/layer (~450 total) vs ~5,500 for the traced graph; attention is one fused kernel on WebGPU.

Inputs : input_ids [1,S] int64 | seqlens_k [1] int32 (= total_len-1) | total_sequence_length [] int32 | past_key_values.{i}.{key,value} [1,4,P,128]
Outputs: logits [1,16385] | stop [1,1] fp32 | present.{i}.{key,value}
Weights are written fp16 (or fp32 with --act fp32 for CPU validation); quantize afterwards with quantize_onnx.py (MatMulNBits).
Full-attention layers have NO rotary (Cohere2), sliding layers use interleaved rotary; sliding window 4096 never matters for our lengths.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, onnx, torch
from onnx import TensorProto as TP, helper as H, numpy_helper as NH
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_decoder import build_from_teacher

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="onnx_fast_fp16/model.onnx")
ap.add_argument("--act", default="fp16", choices=["fp16", "fp32"])
ap.add_argument("--weights-from", default=None)
ap.add_argument("--max-pos", type=int, default=4096)
args = ap.parse_args()
t0 = time.time()
if args.weights_from: os.environ["WEIGHTS_DIR"] = args.weights_from
m, cfg, V = build_from_teacher(torch.float32)
NL, NH_, NKV, HD, I, EPS = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"], cfg["layer_norm_eps"]
HID = cfg["hidden_size"]
A = TP.FLOAT16 if args.act == "fp16" else TP.FLOAT
npA = np.float16 if args.act == "fp16" else np.float32
types = cfg["layer_types"]
nodes, inits = [], []
def init(name, arr, dt=None): 
    inits.append(NH.from_array(np.ascontiguousarray(arr).astype(dt or npA), name)); return name
def node(op, ins, outs, domain="", **attrs): nodes.append(H.make_node(op, ins, outs, name=outs[0] + "_n", domain=domain, **attrs)); return outs[0]
def cast(x, to, name): return node("Cast", [x], [name], to=to) if args.act == "fp16" else x   # only needed for the fp16 activation build
def T(x): return x.detach().cpu().numpy()

# rotary caches [max_pos, 64]
inv = 1.0 / (cfg["rope_theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
ang = np.arange(args.max_pos, dtype=np.float64)[:, None] * inv[None, :]
init("cos_cache", np.cos(ang)); init("sin_cache", np.sin(ang))
init("embed", T(m.embed)); init("head", T(m.out_rows).T)            # head stored [2048,16385]
init("final_norm.w", T(m.norm.weight), np.float32)

# ---- graph inputs
ins = [H.make_tensor_value_info("input_ids", TP.INT64, [1, "S"]), H.make_tensor_value_info("seqlens_k", TP.INT32, [1]),
       H.make_tensor_value_info("total_sequence_length", TP.INT32, [])]
outs = [H.make_tensor_value_info("logits", A, [1, 16385]), H.make_tensor_value_info("stop", TP.FLOAT, [1, 1])]
for i in range(NL):
    for kv in ("key", "value"):
        ins.append(H.make_tensor_value_info(f"past_key_values.{i}.{kv}", A, [1, NKV, "P", HD]))
        outs.append(H.make_tensor_value_info(f"present.{i}.{kv}", A, [1, NKV, "PT", HD]))

# ---- embedding -> fp32 residual
x = node("Gather", ["embed", "input_ids"], ["emb"], axis=0)
x = node("Cast", [x], ["x_0"], to=TP.FLOAT) if args.act == "fp16" else x
for i, L in enumerate(m.layers):
    p = f"layers.{i}."
    init(p + "ln.w", T(L.ln.weight), np.float32)
    init(p + "qkv", np.concatenate([T(L.q), T(L.k), T(L.v)], axis=1))              # [2048, 3072]
    init(p + "o", T(L.o))
    init(p + "gateup", np.concatenate([T(L.gate), T(L.up)], axis=1))              # [2048, 22016]
    init(p + "down", T(L.down)); 
    if i == 0: pass
    h = node("LayerNormalization", [x, p + "ln.w"], [p + "h32"], axis=-1, epsilon=EPS, stash_type=1)
    h = cast(h, TP.FLOAT16, p + "h")
    qkv = node("MatMul", [h, p + "qkv"], [p + "qkv_out"])
    rope = types[i] == "sliding_attention"
    gqa_in = [qkv, "", "", f"past_key_values.{i}.key", f"past_key_values.{i}.value", "seqlens_k", "total_sequence_length"] + ([("cos_cache"), ("sin_cache")] if rope else [])
    attn = [p + "attn", f"present.{i}.key", f"present.{i}.value"]
    nodes.append(H.make_node("GroupQueryAttention", gqa_in, attn, name=p + "gqa", domain="com.microsoft", num_heads=NH_, kv_num_heads=NKV,
                             do_rotary=1 if rope else 0, rotary_interleaved=1, scale=HD ** -0.5, local_window_size=-1))
    ao = node("MatMul", [attn[0], p + "o"], [p + "o_out"])
    gu = node("MatMul", [h, p + "gateup"], [p + "gu"])
    init(p + "split", np.array([I, I], dtype=np.int64), np.int64)
    g_, u_ = node("Split", [gu, p + "split"], [p + "g", p + "u"], axis=-1) if False else (None, None)
    nodes.append(H.make_node("Split", [gu, p + "split"], [p + "g", p + "u"], name=p + "split_n", axis=-1))
    sg = node("Sigmoid", [p + "g"], [p + "sg"]); act = node("Mul", [p + "g", sg], [p + "silu"]); act = node("Mul", [act, p + "u"], [p + "act"])
    mo = node("MatMul", [act, p + "down"], [p + "mlp_out"])
    ao32 = node("Cast", [ao], [p + "o32"], to=TP.FLOAT) if args.act == "fp16" else ao
    mo32 = node("Cast", [mo], [p + "m32"], to=TP.FLOAT) if args.act == "fp16" else mo
    s1 = node("Add", [x, ao32], [p + "r1"]); x = node("Add", [s1, mo32], [f"x_{i + 1}"])

# ---- last position, final norm, head, stop head
init("neg1", np.array([-1], dtype=np.int64), np.int64); init("zero", np.array([0], dtype=np.int64), np.int64)
init("big", np.array([2 ** 31 - 1], dtype=np.int64), np.int64); init("ax1", np.array([1], dtype=np.int64), np.int64)
last = node("Slice", [x, "neg1", "big", "ax1"], ["last"])
hn = node("LayerNormalization", [last, "final_norm.w"], ["hn32"], axis=-1, epsilon=EPS, stash_type=1)
hn_a = cast(hn, TP.FLOAT16, "hn")
hn2 = node("Squeeze", [hn_a, "ax1"], ["hn_sq"]) 
node("MatMul", [hn2, "head"], ["logits"])
# stop head (fp32): LayerNorm(2048) -> Linear -> GELU -> Linear -> Sigmoid
st = m.stop
init("st.ln.w", T(st[0].weight), np.float32); init("st.ln.b", T(st[0].bias), np.float32)
init("st.w1", T(st[1].weight).T, np.float32); init("st.b1", T(st[1].bias), np.float32)
init("st.w2", T(st[3].weight).T, np.float32); init("st.b2", T(st[3].bias), np.float32)
hs = node("Squeeze", [hn], ["hs_sq"] , ) if False else None
nodes.append(H.make_node("Squeeze", [hn, "ax1"], ["hs32"], name="hs32_n"))
node("LayerNormalization", ["hs32", "st.ln.w", "st.ln.b"], ["st_ln"], axis=-1, epsilon=1e-5, stash_type=1)
node("MatMul", ["st_ln", "st.w1"], ["st_m1"]); node("Add", ["st_m1", "st.b1"], ["st_a1"])
init("c_half", np.array(0.5, dtype=np.float32), np.float32); init("c_one", np.array(1.0, dtype=np.float32), np.float32); init("c_rs2", np.array(1 / np.sqrt(2), dtype=np.float32), np.float32)
node("Mul", ["st_a1", "c_rs2"], ["st_s"]); node("Erf", ["st_s"], ["st_e"]); node("Add", ["st_e", "c_one"], ["st_e1"]); node("Mul", ["st_a1", "st_e1"], ["st_g0"]); node("Mul", ["st_g0", "c_half"], ["st_g"])
node("MatMul", ["st_g", "st.w2"], ["st_m2"]); node("Add", ["st_m2", "st.b2"], ["st_a2"]); node("Sigmoid", ["st_a2"], ["stop"])

g = H.make_graph(nodes, "ira_decoder", ins, outs, initializer=inits)
model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)], producer_name="ira-fast-graph")
model.ir_version = 9
out = Path(args.out); out.mkdir(parents=True, exist_ok=True) if False else out.parent.mkdir(parents=True, exist_ok=True)
onnx.save_model(model, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=out.name + ".data", size_threshold=1024)
print(f"nodes {len(nodes)} | wrote {out} ({sum(p.stat().st_size for p in out.parent.iterdir())/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)
