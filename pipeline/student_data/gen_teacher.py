"""
Teacher pass: run rumik-oss-1 (NVFP4, vLLM + TinyAya plugin) as speaker Ira over corpus.jsonl and save,
per sample, the generated Mimi token ids plus the teacher's top-K logprobs at every position
(for logit distillation of the small browser student).

Run inside vllm/vllm-openai:v0.20.0 (see run_teacher.sh). Resumable: finished chunks are skipped.

Prompt:  <text>Ira: [<description="tone, accent, pace">] reply<audio>

Output (student_data/teacher/chunk_XXXXX.npz), ragged arrays over the chunk's samples:
    ids          [S]            corpus row ids (str)
    n_samp       [S]            sample index within row (0..n-1)
    prompt_txt   [S]            the exact text sent after 'Ira: '
    tok          [sum_T]        int32 generated token ids (incl. final </audio> if it terminated)
    off          [S+1]          offsets into tok / topk_*
    topk_id      [sum_T, K]     int32 top-K token ids at each step (raw logits, before temperature/top-k sampling)
    topk_lp      [sum_T, K]     float16 their logprobs
    done         [S]            bool, True if generation ended with </audio> (False = hit max_tokens)
"""
import argparse
import json
import re
import time
import wave
from pathlib import Path

import numpy as np
import torch

AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")


def build_text(row):
    d = row.get("description")
    return f'<description="{d}"> {row["reply"]}' if d else row["reply"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/workspace/student_data/corpus.jsonl")
    ap.add_argument("--repo-dir", default="/models/rumik-oss-1-nvfp4-vllm")
    ap.add_argument("--out-dir", default="/workspace/student_data/teacher")
    ap.add_argument("--speaker", default="Ira")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=1, help="independent samples per text (prosody diversity)")
    ap.add_argument("--chunk", type=int, default=256, help="texts per generate() call / saved chunk")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--save-wavs", type=int, default=0, help="decode first N samples to wav for listening")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel
    from vllm import LLM, SamplingParams

    repo = Path(args.repo_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.corpus)]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    vocab = tok.get_vocab()
    end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}

    llm = LLM(model=str(repo), trust_remote_code=True, quantization="modelopt_fp4", dtype="bfloat16",
              gpu_memory_utilization=0.5, max_model_len=4096, seed=args.seed)
    params = SamplingParams(temperature=0.8, top_k=30, max_tokens=args.max_tokens, stop_token_ids=[end_id],
                            n=args.n_samples, logprobs=args.topk, seed=args.seed)

    mimi = sr = None
    if args.save_wavs:
        codec = repo / "codec"
        mimi = MimiModel.from_pretrained(codec, dtype=torch.bfloat16).eval().to("cuda")
        sr = int(AutoFeatureExtractor.from_pretrained(codec).sampling_rate)
        (out / "wavs").mkdir(exist_ok=True)

    def to_wav(token_ids, path):
        values, frame, exp = [], [], 0
        for t in token_ids:
            it = mapping.get(int(t))
            if it is None:
                frame, exp = [], 0
                continue
            c, q = it
            if q == exp:
                frame.append(c); exp += 1
                if exp == 8:
                    values.extend(frame); frame, exp = [], 0
            elif q == 0:
                frame, exp = [c], 1
            else:
                frame, exp = [], 0
        if not values:
            return 0.0
        with torch.inference_mode():
            codes = torch.tensor(values, device="cuda").reshape(1, -1, 8).transpose(1, 2)
            a = mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((a.numpy() * 32767).astype("<i2").tobytes())
        return a.shape[-1] / sr

    K = args.topk
    wavs_left = args.save_wavs
    t_all = time.perf_counter()
    total_tok = 0
    for ci in range(0, len(rows), args.chunk):
        path = out / f"chunk_{ci // args.chunk:05d}.npz"
        if path.exists():
            print(f"[skip] {path.name}", flush=True)
            continue
        chunk = rows[ci : ci + args.chunk]
        texts = [build_text(r) for r in chunk]
        prompts = [f"<text>{args.speaker}: {t}<audio>" for t in texts]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        wall = time.perf_counter() - t0

        ids, nsamp, ptxt, done, toks, offs, tid, tlp = [], [], [], [], [], [0], [], []
        for r, txt, o in zip(chunk, texts, outs):
            for si, c in enumerate(o.outputs):
                T = len(c.token_ids)
                ids.append(r["id"]); nsamp.append(si); ptxt.append(txt)
                done.append(bool(c.token_ids) and int(c.token_ids[-1]) == end_id)
                toks.extend(int(x) for x in c.token_ids)
                offs.append(offs[-1] + T)
                for lp in c.logprobs:
                    items = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:K]
                    row_id = [k for k, _ in items] + [-1] * (K - len(items))
                    row_lp = [v.logprob for _, v in items] + [-1e4] * (K - len(items))
                    tid.append(row_id); tlp.append(row_lp)
                if wavs_left > 0:
                    dur = to_wav(c.token_ids, out / "wavs" / f"{r['id']}_{si}.wav")
                    wavs_left -= 1
                    print(f"   wav {r['id']}_{si}: {dur:.1f}s for {len(txt.split())} words | {txt[:70]}", flush=True)
        np.savez(path, ids=np.array(ids), n_samp=np.array(nsamp, dtype=np.int16), prompt_txt=np.array(ptxt),
                 tok=np.array(toks, dtype=np.int32), off=np.array(offs, dtype=np.int64),
                 topk_id=np.array(tid, dtype=np.int32).reshape(-1, K), topk_lp=np.array(tlp, dtype=np.float16).reshape(-1, K),
                 done=np.array(done))
        total_tok += len(toks)
        print(f"[{path.name}] {len(ids)} samples, {len(toks)} tok, {len(toks)/wall:.0f} tok/s, "
              f"unterminated={sum(not d for d in done)}, elapsed={time.perf_counter()-t_all:.0f}s", flush=True)
    print(f"done: {total_tok} tokens in {time.perf_counter()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
