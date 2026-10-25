"""
Shared pieces for the Ira-student: data access, batching, and the distillation loss.

Sequence for one sample (student ids):  prompt (BOS <text> ...text... <audio>) + audio tokens o[0..T-2]
Targets:  out index o[0..T-1]  (0 = </audio>, 1+a = Mimi token a), predicted at positions P-1 .. P+T-2.
Teacher signal: top-K logprobs (out-index space) at each of those T positions, restricted to audio + </audio>
(the teacher's stop head forces </audio> at the last position, so p(</audio>)=1 there).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent / "student"
N_OUT = 16385  # </audio> + 16384 audio tokens


class Data:
    def __init__(self, root=ROOT):
        d = np.load(root / "data.npz")
        self.ptok, self.poff = d["ptok"], d["poff"]
        self.otok, self.ooff = d["otok"], d["ooff"]
        self.tk_id, self.tk_lp = d["tk_id"], d["tk_lp"]
        self.is_val, self.ids = d["is_val"], d["ids"]
        self.vocab = json.load(open(root / "vocab.json"))
        self.audio_base = self.vocab["audio_base"]
        self.n_vocab = self.vocab["n_vocab"]
        self.decomp = {int(k): v for k, v in self.vocab["decomp"].items()}
        self.train_idx = np.where(~self.is_val)[0]
        self.val_idx = np.where(self.is_val)[0]
        self.out_rows = torch.tensor([4] + list(range(self.audio_base, self.audio_base + 16384)))  # out index -> embed row

    def length(self, i):
        return (self.poff[i + 1] - self.poff[i]) + (self.ooff[i + 1] - self.ooff[i]) - 1

    def sample(self, i, rng=None, aug_prob=0.0):
        p = self.ptok[self.poff[i]: self.poff[i + 1]].tolist()
        if rng is not None and aug_prob > 0:  # robustness: randomly spell some multi-byte tokens out as bytes
            q = []
            for t in p:
                if t in self.decomp and rng.random() < aug_prob:
                    q.extend(self.decomp[t])
                else:
                    q.append(t)
            p = q
        o = self.otok[self.ooff[i]: self.ooff[i + 1]]
        a = np.where(o[:-1] == 0, 4, o[:-1] - 1 + self.audio_base)  # inputs: o[0..T-2] as embed rows
        ids = np.concatenate([np.array(p, np.int64), a.astype(np.int64)])
        return ids, len(p), o, self.tk_id[self.ooff[i]: self.ooff[i + 1]], self.tk_lp[self.ooff[i]: self.ooff[i + 1]]

    def batches(self, idx, max_tokens, rng, aug_prob=0.0, shuffle=True):
        """Length-bucketed batches under a padded-token budget. Yields dict of tensors."""
        idx = np.array(idx)
        if shuffle:
            idx = idx[rng.permutation(len(idx))]
        chunks = [idx[k:k + 512] for k in range(0, len(idx), 512)]  # sort within windows -> similar lengths
        groups = []
        for c in chunks:
            c = sorted(c, key=self.length)
            cur, mx = [], 0
            for i in c:
                L = self.length(i)
                if cur and max(mx, L) * (len(cur) + 1) > max_tokens:
                    groups.append(cur); cur, mx = [], 0
                cur.append(i); mx = max(mx, L)
            if cur:
                groups.append(cur)
        if shuffle:
            groups = [groups[j] for j in rng.permutation(len(groups))]
        for g in groups:
            yield self.collate(g, rng, aug_prob)

    def collate(self, g, rng=None, aug_prob=0.0):
        items = [self.sample(i, rng, aug_prob) for i in g]
        B, L = len(items), max(len(x[0]) for x in items)
        ids = torch.zeros(B, L, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        pos, tgt, tkid, tklp, bidx = [], [], [], [], []
        for b, (x, P, o, ti, tl) in enumerate(items):
            ids[b, :len(x)] = torch.from_numpy(x); attn[b, :len(x)] = 1
            T = len(o)
            pos.append(torch.arange(P - 1, P - 1 + T)); bidx.append(torch.full((T,), b))
            tgt.append(torch.from_numpy(o.astype(np.int64)))
            tkid.append(torch.from_numpy(ti.astype(np.int64))); tklp.append(torch.from_numpy(tl.astype(np.float32)))
        return dict(ids=ids, attn=attn, bidx=torch.cat(bidx), pos=torch.cat(pos), tgt=torch.cat(tgt),
                    tkid=torch.cat(tkid), tklp=torch.cat(tklp))


def _chunk_loss(h, w, tgt, tkid, tklp, ce_w, stop_w):
    """h [n,H] hidden at prediction positions, w [N_OUT,H] output rows. Returns (loss_sum, weight_sum, kd_sum, ce_sum)."""
    logq = F.log_softmax((h @ w.t()).float(), dim=-1)
    valid = tkid >= 0
    lp = tklp.masked_fill(~valid, -1e4)
    pt = torch.softmax(lp, dim=-1) * valid  # top-K renormalised teacher probs
    kd = -(pt * logq.gather(1, tkid.clamp(min=0))).sum(-1)
    ce = -logq.gather(1, tgt[:, None]).squeeze(1)
    wt = torch.where(tgt == 0, torch.full_like(kd, stop_w), torch.ones_like(kd))  # </audio> position up-weighted
    return ((kd + ce_w * ce) * wt).sum(), wt.sum(), (kd * wt).sum(), (ce * wt).sum()


def distill_loss(model, batch, out_rows, ce_w=0.2, stop_w=20.0, chunk=4096, device="cuda", dtype=torch.bfloat16):
    ids, attn = batch["ids"].to(device), batch["attn"].to(device)
    with torch.autocast("cuda", dtype=dtype):
        hid = model.model(input_ids=ids, attention_mask=attn).last_hidden_state
        h = hid[batch["bidx"].to(device), batch["pos"].to(device)]
        w = model.get_input_embeddings().weight[out_rows.to(device)]
        tgt, tkid, tklp = batch["tgt"].to(device), batch["tkid"].to(device), batch["tklp"].to(device)
        tot = wsum = kd = ce = 0.0
        for s in range(0, h.shape[0], chunk):
            l, ws, k, c = checkpoint(_chunk_loss, h[s:s + chunk], w, tgt[s:s + chunk], tkid[s:s + chunk],
                                     tklp[s:s + chunk], ce_w, stop_w, use_reentrant=False)
            tot, wsum, kd, ce = tot + l, wsum + ws, kd + k, ce + c
    return tot / wsum, (kd / wsum).detach(), (ce / wsum).detach()
