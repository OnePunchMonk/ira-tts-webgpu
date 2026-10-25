"""Sample student (3 seeds) and teacher (1) on given texts; save wavs + Whisper transcripts."""
import sys, json, wave, re
from pathlib import Path
import numpy as np, torch
from scipy.signal import resample_poly
from transformers import AutoTokenizer, Cohere2ForCausalLM, LogitsProcessor, LogitsProcessorList, MimiModel, AutoFeatureExtractor, pipeline
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))
from student_lib import Data
import infer as teacher_infer
ckpt, outdir = sys.argv[1], R / sys.argv[2]; texts = json.loads(sys.argv[3]); outdir.mkdir(exist_ok=True)
data = Data(); V = data.vocab
tok = AutoTokenizer.from_pretrained(R.parent / "rumik-oss-1", trust_remote_code=True)
tv = tok.get_vocab(); inv = {i: s for s, i in tv.items()}
t2s = {t: V["text_base"] + j for j, t in enumerate(V["teacher_text_ids"])}
special = {int(k): v for k, v in V["teacher_special_ids"].items()}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
def pids(txt):
    ids = [1]
    for t in tok("<text>" + txt + "<audio>", add_special_tokens=False).input_ids:
        if t in special: ids.append(special[t])
        elif t in t2s: ids.append(t2s[t])
        else: ids.extend(V["byte_sid"][str(u2b[ch])] for ch in inv[t])
    return ids
class AudioOnly(LogitsProcessor):
    def __init__(s, rows, nv): s.m = torch.full((nv,), float("-inf")); s.m[rows] = 0.0
    def __call__(s, i, sc): return sc + s.m.to(sc.device, sc.dtype)
def save(path, a, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
model = Cohere2ForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()
lp = LogitsProcessorList([AudioOnly(data.out_rows, V["n_vocab"])])
mimi = MimiModel.from_pretrained(R.parent / "rumik-oss-1" / "codec", dtype=torch.bfloat16).eval().cuda()
sr = int(AutoFeatureExtractor.from_pretrained(R.parent / "rumik-oss-1" / "codec").sampling_rate)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
rows = []
def log(name, a, lang):
    save(outdir / f"{name}.wav", a, sr)
    h = asr({"raw": resample_poly(a, 16000, sr).astype(np.float32), "sampling_rate": 16000}, generate_kwargs={"language": lang, "task": "transcribe"})["text"].strip()
    rows.append((name, round(len(a) / sr, 1), h)); print(f"{name:26s} {len(a)/sr:5.1f}s  whisper: {h}", flush=True)
for ti, (label, text, lang) in enumerate(texts):
    for seed in range(3):
        torch.manual_seed(seed)
        ids = torch.tensor([pids(text)]).cuda()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.8, top_k=30,
                                 max_new_tokens=900, eos_token_id=4, pad_token_id=0, logits_processor=lp)[0, ids.shape[1]:].cpu().numpy()
        e = np.where(out == 4)[0]; out = out[: e[0]] if len(e) else out
        a = out - V["audio_base"]; a = a[(a >= 0) & (a < 16384)]; nfr = len(a) // 8
        vals = torch.tensor(a[: nfr * 8].reshape(nfr, 8) // 8).cuda()
        with torch.inference_mode(): au = mimi.decode(vals.T[None]).audio_values[0, 0].float().cpu().numpy()
        log(f"student_{label}_s{seed}", au, lang)
del model; torch.cuda.empty_cache()
T = teacher_infer.RumikTTS(R.parent / "rumik-oss-1", "cuda", torch.bfloat16)
for label, text, lang in texts:
    torch.manual_seed(0)
    au, st = T.synthesize(text, speaker="Ira", max_new_tokens=900)
    log(f"teacher_{label}", au.numpy() if hasattr(au, "numpy") else np.asarray(au), lang)
json.dump(rows, open(outdir / "transcripts.json", "w"), ensure_ascii=False, indent=1)
