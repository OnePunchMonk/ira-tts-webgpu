// Pure logic for the Ira browser TTS demo (no DOM / no onnxruntime) so it can be unit-tested in Node.
//
// Model I/O recap (see rumik-oss-model-dissection/web_tts/ref_decoder.py):
//   decoder input ids live in a pruned vocabulary: 16 specials | 32,000 text tokens | 16,384 audio tokens (audio_base + a, a = value*8 + quantizer)
//   decoder output = 16,385 logits: index 0 = </audio>, index 1 + a = audio token a (the teacher's restricted distribution)
//   one Mimi frame = 8 consecutive audio tokens with quantizers 0..7 in order; 12.5 frames/s -> 100 tokens/s of audio.

export const FRAME_TOKENS = 8;
export const TOKENS_PER_SECOND = 100;
export const SAMPLE_RATE = 24000;
export const N_OUT = 16385;

// teacher special-token ids -> pruned ids (BOS, <text>, <audio>, </audio>, <laugh>, <chuckle>, <sigh>)
const TEACHER_SPECIAL = new Map([[2, 1], [277392, 2], [277393, 3], [277394, 4], [277401, 5], [277402, 6], [277403, 7]]);

// GPT-2 style byte <-> unicode table used by byte-level BPE vocabularies
function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const u2b = new Map();
  bs.forEach((b, i) => u2b.set(String.fromCodePoint(cs[i]), b));
  return u2b;
}
const U2B = bytesToUnicode();

export function buildVocab(webVocab) {
  const t2s = new Map(webVocab.teacher_text_ids.map((t, j) => [t, webVocab.text_base + j]));
  return { t2s, audioBase: webVocab.audio_base, nVocab: webVocab.n_vocab, byteSid: webVocab.byte_sid };
}

/** encoded = tokenizer.encode(prompt) -> {ids, tokens}; returns pruned input ids (rare tokens are spelled out as byte tokens). */
export function toPrunedIds(encoded, vocab) {
  const out = [];
  let oov = 0;
  for (let i = 0; i < encoded.ids.length; i++) {
    const id = encoded.ids[i];
    if (TEACHER_SPECIAL.has(id)) out.push(TEACHER_SPECIAL.get(id));
    else if (vocab.t2s.has(id)) out.push(vocab.t2s.get(id));
    else {
      oov++;
      for (const ch of encoded.tokens[i]) {
        const b = U2B.get(ch);
        if (b === undefined) throw new Error(`cannot spell out token ${JSON.stringify(encoded.tokens[i])}`);
        out.push(vocab.byteSid[String(b)]);
      }
    }
  }
  return { ids: out, oov };
}

/** The exact text the teacher was prompted with. `description` like "happy, Hindi accent, steady pace" (or empty). */
export function buildPrompt(text, description) {
  const body = description ? `<description="${description}"> ${text}` : text;
  return `<text>Ira: ${body}<audio>`;
}

// ---- half precision ----
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let u = 0; u < 65536; u++) {
    const s = (u >> 15) & 1, e = (u >> 10) & 31, f = u & 1023;
    t[u] = e === 0 ? (s ? -1 : 1) * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s ? -Infinity : Infinity) : (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();
export function toFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) { const o = new Float32Array(data.length); for (let i = 0; i < data.length; i++) o[i] = F16[data[i]]; return o; }
  return Float32Array.from(data); // Float16Array (newer browsers) or plain array
}
export function fromFloat32ToHalfBits(x) { // only used by tests / dummy inputs
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  f32[0] = x; const v = u32[0], s = (v >> 16) & 0x8000; let e = ((v >> 23) & 0xff) - 112, m = (v >> 13) & 0x3ff;
  if (e <= 0) return s; if (e >= 31) return s | 0x7c00; return s | (e << 10) | m;
}

// ---- sampling ----
/** mulberry32 PRNG so runs can be seeded. */
export function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

/**
 * Sample the next audio token. `step` = number of audio tokens generated so far. Only tokens whose quantizer equals step % 8 are
 * allowed (valid frame structure); </audio> (index 0) is only allowed at a frame boundary when allowEnd.
 * Returns the output index (0 = end, 1 + a = audio token a).
 */
export function sampleToken(logits, step, { temperature = 0.8, topK = 30, allowEnd = false, rand = Math.random } = {}) {
  const q = step % FRAME_TOKENS;
  const cand = [], val = [];
  for (let a = q; a < 16384; a += FRAME_TOKENS) { cand.push(a + 1); val.push(logits[a + 1]); }
  if (allowEnd && q === 0) { cand.push(0); val.push(logits[0]); }
  const order = val.map((_, i) => i).sort((i, j) => val[j] - val[i]).slice(0, topK);
  const m = val[order[0]];
  const w = order.map((i) => Math.exp((val[i] - m) / temperature));
  const z = w.reduce((a, b) => a + b, 0);
  let r = rand() * z;
  for (let k = 0; k < order.length; k++) { r -= w[k]; if (r <= 0) return cand[order[k]]; }
  return cand[order[order.length - 1]];
}

/** output index -> decoder input row (pruned vocab) */
export const outToInputId = (o, audioBase) => (o === 0 ? 4 : audioBase + (o - 1));

/** audio token list (a = value*8+q, length multiple of 8) -> Int64 codes [1, 32, T] for the 32-codebook Mimi ONNX (rows 8..31 are zero:
 *  their embedding tables are zeroed in decoder_8cb_fp32.onnx so they contribute nothing). */
export function mimiCodes(tokens) {
  const T = Math.floor(tokens.length / FRAME_TOKENS);
  const codes = new BigInt64Array(32 * T);
  for (let t = 0; t < T; t++) for (let q = 0; q < FRAME_TOKENS; q++) codes[q * T + t] = BigInt(tokens[t * FRAME_TOKENS + q] >> 3);
  return { codes, T };
}

/** Float32 mono -> 16-bit PCM WAV bytes */
export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Uint8Array(buf);
}

/** Split text into sentences for pipelined synthesis (keeps the sentence-final punctuation; handles the Devanagari danda). */
export function splitSentences(text, minLen = 12) {
  const parts = text.replace(/\s+/g, " ").trim().match(/[^.!?।\n]+[.!?।]*\s*/g) || [text];
  const out = [];
  for (const p of parts) {
    const s = p.trim();
    if (!s) continue;
    if (out.length && (out[out.length - 1].length < minLen)) out[out.length - 1] += " " + s; else out.push(s);
  }
  return out.length ? out : [text.trim()];
}

/** Longest audio we allow for a piece of text (guards against runaway generations): ~1.3 s/word + 2 s (teacher median 0.45 s/word, p99 0.9). */
export function maxSecondsFor(text) {
  const words = (text.replace(/<[^>]+>/g, " ").match(/\S+/g) || []).length;
  return Math.min(30, 1.3 * words + 2);
}
