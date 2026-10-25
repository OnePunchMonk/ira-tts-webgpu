"""
Merge subagent shard outputs (shards/out_XX.jsonl) with their prompts into replies.jsonl,
dropping skipped / malformed / non-speakable replies. Prints per-reason drop counts.

Output: student_data/replies.jsonl  {"id","lang","prompt","reply"}
"""
import glob
import json
import re
from collections import Counter

DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")
BAD = re.compile(r"[*#_`>\[\]{}|()(（]|https?://|[\U0001F300-\U0001FAFF]")
TAGS = re.compile(r"<(laugh|chuckle|sigh)>")
OTHER_TAG = re.compile(r"<(?!/?(laugh|chuckle|sigh)>)[^>]*>")


# replies where Ira declines/hedges about an unknown entity; capped so the student doesn't learn to hedge by default
HEDGE = re.compile(
    r"not sure|not certain|don't (really )?(know|have)|do not (know|have)|पक्की (जानकारी|याद)|पक्का (पता|नहीं)|"
    r"जानकारी नहीं|यक़ीन नहीं|पता नहीं|गढ़|invent|make (it|things) up|कोई जानकारी नहीं|ज़्यादा जानकारी नहीं",
    re.I,
)
MAX_HEDGE_FRAC = 0.12


def cap_hedges(kept, drops, seed=0):
    import random

    rnd = random.Random(seed)
    hedged = [r for r in kept if HEDGE.search(r["reply"])]
    plain = [r for r in kept if not HEDGE.search(r["reply"])]
    allow = int(MAX_HEDGE_FRAC * len(plain) / (1 - MAX_HEDGE_FRAC))
    rnd.shuffle(hedged)
    drops["hedge_cap"] += max(0, len(hedged) - allow)
    out = plain + hedged[:allow]
    rnd.shuffle(out)
    return out


def main():
    prompts = {}
    for f in glob.glob("student_data/shards/in_*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            prompts[d["id"]] = d
    kept, drops, langs = [], Counter(), Counter()
    seen = set()
    for f in sorted(glob.glob("student_data/shards/out_*.jsonl")):
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                drops["bad_json"] += 1
                continue
            p = prompts.get(d.get("id"))
            if p is None:
                drops["unknown_id"] += 1
                continue
            if d.get("skip"):
                drops["skipped"] += 1
                continue
            r = (d.get("reply") or "").strip()
            r = re.sub(r"\s+", " ", r)
            n = len(r.split())
            if d["id"] in seen:
                drops["dup"] += 1
            elif not r:
                drops["empty"] += 1
            elif n < 6 or n > 70:
                drops["length"] += 1
            elif BAD.search(r) or OTHER_TAG.search(r):
                drops["markup"] += 1
            elif len(TAGS.findall(r)) > 1:
                drops["multi_tag"] += 1
            elif p["lang"] in ("hi", "mixed") and not DEV.search(r):
                drops["no_devanagari"] += 1
            elif p["lang"] == "en" and DEV.search(r) and not re.search(r"hindi|हिंदी|हिन्दी", p["prompt"], re.I):
                drops["en_prompt_hindi_reply"] += 1
            else:
                seen.add(d["id"])
                langs[p["lang"]] += 1
                kept.append({"id": d["id"], "lang": p["lang"], "prompt": p["prompt"], "reply": r})
    kept = cap_hedges(kept, drops)
    langs = Counter(r["lang"] for r in kept)
    with open("student_data/replies.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} / {len(prompts)}  by lang: {dict(langs)}")
    print("dropped:", dict(drops))
    words = sorted(len(r["reply"].split()) for r in kept)
    if words:
        print(f"reply words p10/p50/p90: {words[len(words)//10]}/{words[len(words)//2]}/{words[9*len(words)//10]}")


if __name__ == "__main__":
    main()
