"""
Vocabulary for the browser TEACHER model: the teacher is prompted as "<text>Ira: {desc?} text<audio>" so tokens are counted on that
exact form. Kept set = 256 byte tokens + every token seen in our corpora (count >= 1) + the most frequent tokens of ~30k Sarvam
conversations (broad Hindi/English coverage) up to --n-text. Anything else is spelled out as byte tokens at inference.
Layout identical to student_data/student/vocab.json:  16 specials | text tokens | audio tokens (aligned to 8).
"""
import argparse, json, re
from collections import Counter
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "student_data"
AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")
ap = argparse.ArgumentParser(); ap.add_argument("--n-text", type=int, default=32000); ap.add_argument("--convs", type=int, default=30000)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True)
vocab = tok.get_vocab(); inv = {i: s for s, i in vocab.items()}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
b2u = {b: chr(c) for b, c in zip(bs, cs)}; u2b = {u: b for b, u in b2u.items()}
byte_tids = [vocab[b2u[b]] for b in range(256)]
special_t = {2: 1, 277392: 2, 277393: 3, 277394: 4, 277401: 5, 277402: 6, 277403: 7}
audio = {i for s, i in vocab.items() if AUDIO_RE.match(s)}
banned = audio | set(special_t) | {3, 6}
cnt_corpus = Counter()
for fn in ("corpus.jsonl", "corpus2.jsonl"):
    for l in open(SD / fn):
        d = json.loads(l)
        txt = f'<description="{d["description"]}"> {d["reply"]}' if d.get("description") else d["reply"]
        cnt_corpus.update(tok("<text>Ira: " + txt + "<audio>", add_special_tokens=False).input_ids)
cnt_broad = Counter()
sv = pd.read_parquet(SD / "raw/samvaad/data/train-00000-of-00001.parquet")
for msgs in sv.messages[: args.convs]:
    for m in msgs:
        cnt_broad.update(tok(m["content"][:1500], add_special_tokens=False).input_ids)
keep = list(dict.fromkeys(byte_tids))
seen = set(keep)
for t, _ in cnt_corpus.most_common():
    if t not in banned and t not in seen: keep.append(t); seen.add(t)
n_corpus = len(keep)
for t, _ in cnt_broad.most_common():
    if len(keep) >= args.n_text: break
    if t not in banned and t not in seen: keep.append(t); seen.add(t)
t2s = {t: 16 + j for j, t in enumerate(keep)}
audio_base = 16 + len(keep); audio_base += (-audio_base) % 8
byte_sid = {str(b): t2s[byte_tids[b]] for b in range(256)}
json.dump(dict(teacher_text_ids=keep, text_base=16, audio_base=audio_base, n_vocab=audio_base + 16384,
               specials={"pad": 0, "bos": 1, "text": 2, "audio": 3, "end": 4, "laugh": 5, "chuckle": 6, "sigh": 7},
               teacher_special_ids={str(t): s for t, s in special_t.items()}, byte_sid=byte_sid), open(Path(__file__).parent / "web_vocab.json", "w"))
tot = sum(cnt_broad.values()); cov = sum(c for t, c in cnt_broad.items() if t in seen or t in banned) / tot
print(f"kept {len(keep)} text tokens (bytes 256, corpus {n_corpus}, broad-frequency {len(keep)-n_corpus}) | n_vocab {audio_base+16384} | "
      f"coverage of broad Sarvam text: {100*cov:.2f}% of tokens")
