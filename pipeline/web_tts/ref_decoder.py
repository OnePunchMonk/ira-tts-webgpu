"""
Minimal re-implementation of the rumik-oss-1 (TinyAya / Cohere2) decoder for ONNX export:

  * explicit KV cache in/out, batch size 1, no sliding-window masking (valid while total length < 4096; sliding_window=4096)
  * pruned vocabulary: 48,400 input rows (16 specials + 32,000 text tokens + 16,384 audio tokens, see web_tts/web_vocab.json)
    and a 16,385-way output (</audio> + audio tokens) -> the same restricted distribution the teacher samples from
  * stop head (LayerNorm -> Linear -> GELU -> Linear -> sigmoid) fused into the graph; returns p(stop) for the last position
  * exact Cohere2 math: mean-centred LayerNorm without bias, GPT-J (interleaved) RoPE ONLY on sliding-attention layers
    (full-attention layers use no positional encoding), GQA 16/4, parallel block  x + attn(LN x) + mlp(LN x), logit_scale 1.0

verify_parity() compares against the Hugging Face teacher on real samples.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
VOCAB = Path(__file__).resolve().parent / "web_vocab.json"
TEACHER = ROOT / "rumik-oss-1"
T_AUDIO0 = 261008
SPECIAL_TEACHER = {0: 0, 1: 2, 2: 277392, 3: 277393, 4: 277394, 5: 277401, 6: 277402, 7: 277403}  # row -> teacher id


class LN(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        x = x.float()
        x = x - x.mean(-1, keepdim=True)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight.float() * x


class Layer(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        H, nh, nkv, hd, I = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
        self.nh, self.nkv, self.hd, self.rope = nh, nkv, hd, rope
        # rotate_half for interleaved RoPE as a constant gather + sign (export-friendly): [-x1, x0, -x3, x2, ...]
        self.register_buffer("perm", torch.arange(hd) ^ 1, persistent=False)
        self.register_buffer("sign", torch.tensor([-1.0, 1.0] * (hd // 2)), persistent=False)
        self.ln = LN(H, cfg["layer_norm_eps"])
        # weights stored [in, out] so the graph is a plain MatMul(x, W) with a constant B (what MatMulNBits quantisation expects)
        self.q = nn.Parameter(torch.empty(H, nh * hd)); self.k = nn.Parameter(torch.empty(H, nkv * hd))
        self.v = nn.Parameter(torch.empty(H, nkv * hd)); self.o = nn.Parameter(torch.empty(nh * hd, H))
        self.gate = nn.Parameter(torch.empty(H, I)); self.up = nn.Parameter(torch.empty(H, I)); self.down = nn.Parameter(torch.empty(I, H))

    def forward(self, x, cos, sin, mask, pk, pv):
        B, T, _ = x.shape
        h = self.ln(x).to(x.dtype)
        q = torch.matmul(h, self.q).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = torch.matmul(h, self.k).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = torch.matmul(h, self.v).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.rope:
            def rot(t):
                t = t.float()
                return (t * cos + torch.index_select(t, -1, self.perm) * self.sign * sin).to(x.dtype)
            q, k = rot(q), rot(k)
        k = torch.cat([pk, k], 2); v = torch.cat([pv, v], 2)
        rep = self.nh // self.nkv
        kk = k[:, :, None].expand(B, self.nkv, rep, k.shape[2], self.hd).reshape(B, self.nh, k.shape[2], self.hd)
        vv = v[:, :, None].expand(B, self.nkv, rep, v.shape[2], self.hd).reshape(B, self.nh, v.shape[2], self.hd)
        att = torch.matmul(q, kk.transpose(2, 3)) * (self.hd ** -0.5)
        att = torch.softmax(att.float() + mask, dim=-1).to(x.dtype)
        a = torch.matmul(att, vv).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        attn_out = torch.matmul(a, self.o)
        mlp_out = torch.matmul(F.silu(torch.matmul(h, self.gate)) * torch.matmul(h, self.up), self.down)
        return x + attn_out + mlp_out, k, v


class WebDecoder(nn.Module):
    def __init__(self, cfg, n_vocab):
        super().__init__()
        self.cfg = cfg
        H = cfg["hidden_size"]
        types = cfg["layer_types"]
        self.layers = nn.ModuleList([Layer(cfg, types[i] == "sliding_attention") for i in range(cfg["num_hidden_layers"])])
        self.embed = nn.Parameter(torch.empty(n_vocab, H))
        self.out_rows = nn.Parameter(torch.empty(16385, H))      # </audio> + 16,384 audio rows (separate copy: tied weights in the teacher)
        self.norm = LN(H, cfg["layer_norm_eps"])
        self.stop = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, max(64, H // 4)), nn.GELU(), nn.Linear(max(64, H // 4), 1))
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, cfg["head_dim"], 2).float() / cfg["head_dim"]))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.register_buffer("dup", torch.arange(cfg["head_dim"]) // 2, persistent=False)   # repeat_interleave(.., 2) as a gather

    def forward(self, input_ids, position_ids, *past):
        """input_ids [1,T] int64, position_ids [1,T] int64, past = (k0,v0,k1,v1,...) each [1,4,P,128].
        returns logits [1,16385] (last position), stop [1,1], present (k0,v0,...)"""
        x = F.embedding(input_ids, self.embed)
        T = input_ids.shape[1]; P = past[0].shape[2]
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]              # [1,T,64]
        emb = torch.index_select(freqs, -1, self.dup)                                         # interleaved [f0,f0,f1,f1,...]
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]                                     # [1,1,T,128]
        qpos = (torch.arange(T)[:, None] + P)                                                # absolute index of each query
        kpos = torch.arange(P + T)[None, :]
        mask = torch.where(kpos <= qpos, 0.0, -1e9)[None, None].float()                       # causal
        present = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, mask, past[2 * i], past[2 * i + 1])
            present += [k, v]
        h = self.norm(x).to(x.dtype)[:, -1]                                                   # [1,H]
        logits = torch.matmul(h, self.out_rows.t()) * self.cfg["logit_scale"]
        stop = torch.sigmoid(self.stop(h.float()))
        return (logits, stop, *present)


def build_from_teacher(dtype=torch.float32, src=None):
    """Load teacher weights (HF safetensors; or a QAT-merged export dir via `src` / $WEIGHTS_DIR) into the web decoder with the pruned vocabulary."""
    import os
    from safetensors import safe_open
    src = Path(src or os.environ.get("WEIGHTS_DIR") or TEACHER)
    cfg = json.load(open(TEACHER / "config.json"))
    V = json.load(open(VOCAB))
    m = WebDecoder(cfg, V["n_vocab"]).to(dtype)
    sd = {}
    for f in sorted(list(src.glob("model-*.safetensors")) + list(src.glob("model.safetensors"))):
        with safe_open(str(f.resolve()), "pt") as g:
            for k in g.keys():
                sd[k] = g.get_tensor(k)
    pre = "model."
    E = sd[pre + "embed_tokens.weight"].float()
    with torch.no_grad():
        W = torch.zeros(V["n_vocab"], cfg["hidden_size"])
        for r, t in SPECIAL_TEACHER.items():
            W[r] = E[t]
        W[8:16] = E.mean(0)
        W[V["text_base"]: V["text_base"] + len(V["teacher_text_ids"])] = E[torch.tensor(V["teacher_text_ids"])]
        W[V["audio_base"]: V["audio_base"] + 16384] = E[T_AUDIO0: T_AUDIO0 + 16384]
        m.embed.copy_(W)
        m.out_rows.copy_(torch.cat([E[277394:277395], E[T_AUDIO0: T_AUDIO0 + 16384]]))
        m.norm.weight.copy_(sd[pre + "norm.weight"].float())
        for i, L in enumerate(m.layers):
            p = f"{pre}layers.{i}."
            L.ln.weight.copy_(sd[p + "input_layernorm.weight"].float())
            for name, key in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                              ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
                getattr(L, name).copy_(sd[p + key + ".weight"].float().t())
        for name, key in (("0", "stop_predictor.0"), ("1", "stop_predictor.1"), ("3", "stop_predictor.3")):
            getattr(m.stop, name).weight.copy_(sd[key + ".weight"].float()); getattr(m.stop, name).bias.copy_(sd[key + ".bias"].float())
    m = m.to(dtype)
    m.stop.float()   # the stop head always runs in fp32 (it is fed h.float())
    return m.eval(), cfg, V


def empty_past(cfg, dtype=torch.float32):
    return [torch.zeros(1, cfg["num_key_value_heads"], 0, cfg["head_dim"], dtype=dtype) for _ in range(2 * cfg["num_hidden_layers"])]
