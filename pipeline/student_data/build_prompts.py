"""
Build the user-prompt pool for the Ira-student teacher corpus.

Sources (first user turn only; their assistant replies are NOT used, LFM2 writes replies):
  - sarvamai/samvaad-hi-v1                (Apache-2.0)
  - ai4bharat/indic-instruct-data-v0.1 lm_sys/hi  (real chat prompts, Hindi)

Each prompt is tagged by script: hi (Devanagari), mixed (Devanagari+Latin),
roman (romanized Hinglish, detected via a Hindi stopword list), en.

Output: student_data/prompts.jsonl  {"id","src","lang","text"}
"""
import argparse
import json
import random
import re

import pandas as pd

DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")
BAD = re.compile(r"https?://|```|\{|\}|<[a-z/]|\\n|[0-9]{6,}")
# common romanized-Hindi function words; >=2 hits in a Latin-only prompt => Hinglish
# excludes words that are also common English (me, to, do, ho, hi, in, par, pe, ek, ka, ke ...)
ROMAN_HI = set("""mujhe mujhko kya hai hain nahi nahin aap aapko mera meri mere tum tumhe kaise kaun kab kahan
kyun kyu mein batao bataiye batayein bata karo kijiye hota hoti bhi toh yeh woh isse uske unke chahiye
sakta sakte sakti raha rahi rahe baare jawab dijiye batayiye hum humein lekin aur""".split())


def classify(t: str) -> str:
    d, l = bool(DEV.search(t)), bool(LAT.search(t))
    if d and l:
        return "mixed"
    if d:
        return "hi"
    words = re.findall(r"[a-z]+", t.lower())
    return "roman" if sum(w in ROMAN_HI for w in words) >= 2 else "en"


def ok(t: str, lo: int, hi: int) -> bool:
    return lo <= len(t) <= hi and not BAD.search(t) and "\n" not in t.strip()


def first_user(msgs):
    for m in msgs:
        if m["role"] == "user":
            return m["content"].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="student_data/raw")
    ap.add_argument("--out", default="student_data/prompts.jsonl")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--quota", default="hi=6000,mixed=5000,roman=4000,en=5000",
                    help="max prompts per lang tag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    quota = {k: int(v) for k, v in (kv.split("=") for kv in args.quota.split(","))}

    pool = {k: [] for k in quota}
    sv = pd.read_parquet(f"{args.raw}/samvaad/data/train-00000-of-00001.parquet")
    for i, msgs in enumerate(sv.messages):
        t = first_user(msgs)
        if t and ok(t, args.min_chars, args.max_chars):
            pool[classify(t)].append(("samvaad", f"sv{i}", t))
    lm = pd.read_parquet(f"{args.raw}/indic-instruct/lm_sys/hi-00000-of-00001.parquet")
    for _, row in lm.iterrows():
        t = first_user(row["messages"])
        if t and ok(t, args.min_chars, args.max_chars):
            pool[classify(t)].append(("lm_sys_hi", f"ls{row['id']}", t))

    seen, out = set(), []
    for lang, items in pool.items():
        random.shuffle(items)
        n = 0
        for src, pid, t in items:
            key = re.sub(r"\W+", "", t.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": pid, "src": src, "lang": lang, "text": t})
            n += 1
            if n >= quota[lang]:
                break
        print(f"{lang}: available={len(items)} kept={n}")
    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
