"""
Generate short spoken-style assistant replies with LFM2.5-1.2B-Instruct for each prompt in
prompts.jsonl. These replies are the TEXT side of the teacher corpus (rumik-oss-1 turns them
into Ira audio tokens later).

Output: student_data/replies.jsonl  {"id","lang","prompt","reply"}  (resumable)
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = """You are Ira, a friendly voice assistant. Your reply will be spoken aloud.
Rules:
- Answer in 1 to 3 short sentences, under 45 words total.
- Reply in the same language style as the user: Hindi in Devanagari, English in English, and if the user mixes Hindi and English (or writes Hinglish in Roman letters) reply in natural Hinglish, writing Hindi words in Devanagari and English words in Latin.
- If the user asks you to answer in a specific language, do that.
- No markdown, no lists, no emojis, no URLs, no parentheses. Write numbers as words when short.
- Sound natural and warm, like talking to a friend. If you do not know something, say so briefly."""

BAD = re.compile(r"[*#_`>\[\]{}|]|https?://|[\U0001F300-\U0001FAFF]")


def clean(t: str):
    t = t.strip().replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if not t or BAD.search(t) or len(t.split()) > 70 or len(t) < 8:
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="student_data/prompts.jsonl")
    ap.add_argument("--out", default="student_data/replies.jsonl")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts)]
    if args.n:
        rows = rows[: args.n]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)}
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts to do ({len(done)} already done)")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    kept = dropped = 0
    with open(args.out, "a") as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=140, do_sample=True, temperature=args.temperature,
                    top_p=0.9, repetition_penalty=1.05,
                )
            for r, o in zip(chunk, out):
                reply = clean(tok.decode(o[enc.input_ids.shape[1] :], skip_special_tokens=True))
                if reply is None:
                    dropped += 1
                    continue
                kept += 1
                f.write(json.dumps({"id": r["id"], "lang": r["lang"], "prompt": r["text"], "reply": reply},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + len(chunk)}/{len(rows)}] kept={kept} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
