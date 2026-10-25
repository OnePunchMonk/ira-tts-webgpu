"""
Merge round-2 text-only shards (shards/t_XX.jsonl) into corpus2.jsonl, in the same row format as corpus.jsonl
({"id","src","lang","user","reply","description"}) so gen_teacher.py reads it unchanged.

Checks: id prefix matches file, valid lang, length 3-45 words, markup, Devanagari rules per lang, <=1 inline tag,
description format ('tone, accent, pace'), exact-duplicate text (vs corpus.jsonl and within round 2), and
template near-duplicates (same first 5 words, kept at most MAX_SAME_OPENING times per lang).
"""
import glob
import json
import re
from collections import Counter

DEV = re.compile(r"[ऀ-ॿ]")
BAD = re.compile(r"[*#_`\[\]{}|()（]|https?://|[\U0001F300-\U0001FAFF]")
OTHER_TAG = re.compile(r"<(?!/?(laugh|chuckle|sigh)>)[^>]*>")
TAG = re.compile(r"<(laugh|chuckle|sigh)>")
DESC = re.compile(r"^(happy|sad|angry|excited|professional), (Hindi|Indian English) accent, (slow|fast|steady) pace$")
MAX_SAME_OPENING = 4
norm = lambda t: re.sub(r"\W+", "", TAG.sub("", t).lower())
opening = lambda t: " ".join(re.findall(r"\w+", TAG.sub("", t).lower())[:5])


def main():
    seen = {norm(json.loads(l)["reply"]) for l in open("student_data/corpus.jsonl")}
    drops, rows, open_ct = Counter(), [], Counter()
    for f in sorted(glob.glob("student_data/shards/t_[0-9][0-9].jsonl")):
        tag = re.search(r"t_(\d+)", f).group(1)
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                drops["bad_json"] += 1
                continue
            t = re.sub(r"\s+", " ", (d.get("text") or "").strip())
            lang, desc = d.get("lang"), d.get("description")
            n = len(t.split())
            if isinstance(desc, str) and not desc.endswith(" pace"):
                desc += " pace"
            if not str(d.get("id", "")).startswith(f"t{tag}_"):
                drops["id_prefix"] += 1
            elif lang not in ("hi", "mixed", "en"):
                drops["bad_lang"] += 1
            elif n < 3 or n > 45:
                drops["length"] += 1
            elif BAD.search(t) or OTHER_TAG.search(t):
                drops["markup"] += 1
            elif len(TAG.findall(t)) > 1:
                drops["multi_tag"] += 1
            elif lang in ("hi", "mixed") and not DEV.search(t):
                drops["no_devanagari"] += 1
            elif lang == "en" and DEV.search(t):
                drops["en_has_devanagari"] += 1
            elif desc is not None and not DESC.match(desc):
                drops["bad_description"] += 1
            elif norm(t) in seen:
                drops["dup_text"] += 1
            elif open_ct[(lang, opening(t))] >= MAX_SAME_OPENING:
                drops["same_opening"] += 1
            else:
                seen.add(norm(t)); open_ct[(lang, opening(t))] += 1
                rows.append({"id": d["id"], "src": "t" + tag, "lang": lang, "user": None, "reply": t, "description": desc})
    with open("student_data/corpus2.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(rows)} | dropped {dict(drops)}")
    print("by lang:", dict(Counter(r["lang"] for r in rows)), "| by slice:", dict(sorted(Counter(r["src"] for r in rows).items())))
    print("with description:", sum(r["description"] is not None for r in rows), "| tags:", dict(Counter(m for r in rows for m in TAG.findall(r["reply"]))),
          "| rows with a tag: %.0f%%" % (100 * sum(bool(TAG.search(r["reply"])) for r in rows) / max(1, len(rows))))
    print("tones:", dict(Counter(r["description"].split(",")[0] for r in rows if r["description"])))
    w = sorted(len(r["reply"].split()) for r in rows)
    print(f"words p10/p50/p90: {w[len(w)//10]}/{w[len(w)//2]}/{w[9*len(w)//10]}")


if __name__ == "__main__":
    main()
