"""
Merge conversation shards (shards/conv_XX.jsonl) and the trivia replies (replies.jsonl) into corpus.jsonl.

Checks: id prefix matches file, no duplicate ids / user lines across shards, reply length, markup,
Devanagari presence for hi/mixed/roman replies, valid description format, at most one inline tag.

corpus.jsonl rows: {"id","src","lang","user","reply","description"}  (description null or 'tone, accent, pace')
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
norm = lambda t: re.sub(r"\W+", "", t.lower())


def main():
    rows, drops, seen_id, seen_user = [], Counter(), set(), set()
    for f in sorted(glob.glob("student_data/shards/conv_*.jsonl")):
        tag = re.search(r"conv_(\d+)", f).group(1)
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                drops["bad_json"] += 1
                continue
            u, r, lang, desc = d.get("user", "").strip(), re.sub(r"\s+", " ", d.get("reply", "").strip()), d.get("lang"), d.get("description")
            if tag == "06":  # this shard's tags were script-inserted mid-phrase; drop them rather than teach bad placement
                r = re.sub(r"\s*<(laugh|chuckle|sigh)>\s*", " ", r).strip()
            n = len(r.split())
            if isinstance(desc, str) and not desc.endswith(" pace"):
                desc = desc.strip() + " pace"  # agents wrote 'steady'; Rumik format is 'steady pace'
            if not str(d.get("id", "")).startswith(f"c{tag}_"):
                drops["id_prefix_mismatch"] += 1
            elif d["id"] in seen_id:
                drops["dup_id"] += 1
            elif norm(u) in seen_user:
                drops["dup_user"] += 1
            elif lang not in ("hi", "mixed", "roman", "en") or not u:
                drops["bad_lang_or_empty"] += 1
            elif n < 8 or n > 55:
                drops["length"] += 1
            elif BAD.search(r) or OTHER_TAG.search(r) or BAD.search(u):
                drops["markup"] += 1
            elif len(TAG.findall(r)) > 1 or TAG.search(u):
                drops["tag_misuse"] += 1
            elif lang in ("hi", "mixed", "roman") and not DEV.search(r):
                drops["no_devanagari"] += 1
            elif desc is not None and not DESC.match(desc):
                drops["bad_description"] += 1
            else:
                seen_id.add(d["id"]); seen_user.add(norm(u))
                rows.append({"id": d["id"], "src": "conv", "lang": lang, "user": u, "reply": r, "description": desc})
    nconv = len(rows)
    for l in open("student_data/replies.jsonl"):
        d = json.loads(l)
        if norm(d["prompt"]) in seen_user:
            drops["dup_user"] += 1
            continue
        rows.append({"id": d["id"], "src": "trivia", "lang": d["lang"], "user": d["prompt"], "reply": d["reply"], "description": None})
    with open("student_data/corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"conv kept {nconv}, trivia kept {len(rows)-nconv}, total {len(rows)}")
    print("dropped:", dict(drops))
    print("by lang:", dict(Counter(r["lang"] for r in rows)))
    print("with description:", sum(r["description"] is not None for r in rows),
          "| inline tags:", dict(Counter(m for r in rows for m in TAG.findall(r["reply"]))))
    w = sorted(len(r["reply"].split()) for r in rows)
    print(f"reply words p10/p50/p90: {w[len(w)//10]}/{w[len(w)//2]}/{w[9*len(w)//10]}")


if __name__ == "__main__":
    main()
