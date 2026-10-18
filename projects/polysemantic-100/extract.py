"""Per neuron and pole: the strongest firings plus a random sample from the strong half.

    uv run python projects/polysemantic-100/extract.py      -> data/contexts/<id>.json

Rules (fixed before reading any context):
- Position 0 of each window is skipped (attention-sink token, fires for most neurons).
- Pole +: a_j > 0, pole -: a_j < 0. Pole max = largest |a_j| on that side.
- Strong half = |a_j| >= 0.5 x pole max (Goodfire's cutoff). One firing per window
  (its peak), so one long document cannot fill the list.
- Shown contexts: the top 12 windows by peak, plus 8 random windows from the rest of
  the strong half (seed 0). If the strong half has fewer windows, fill by rank.
- Each context = 40 tokens before the peak token, the peak token marked [[...]], 6 after.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from common import CFG, DATA, load_neurons

N_TOP, N_RAND, BEFORE, AFTER = 12, 8, 40, 6


def load_streams():
    """Neuron-major copies [K, W, T] of the act and gate streams (built once, in chunks)."""
    neurons = load_neurons()
    toks = np.load(DATA / "tokens.npy")
    W, T = toks.shape
    K = len(neurons)
    assert int((DATA / "record_progress.txt").read_text()) == W, "record pass not finished"
    out = []
    for name in ("acts", "gate"):
        nm = DATA / f"{name}_nm.f16"
        if not nm.exists():
            src = np.memmap(DATA / f"{name}.f16", dtype=np.float16, mode="r", shape=(W, T, K))
            dst = np.memmap(nm, dtype=np.float16, mode="w+", shape=(K, W, T))
            for b in range(0, W, 100):
                dst[:, b : b + 100] = np.transpose(np.asarray(src[b : b + 100]), (2, 0, 1))
            dst.flush()
            del dst
        out.append(np.memmap(nm, dtype=np.float16, mode="r", shape=(K, W, T)))
    return neurons, toks, out[0], out[1]


def context(tok, toks, w, p) -> str:
    row = toks[w]
    pre = tok.decode(row[max(0, p - BEFORE) : p].tolist())
    mid = tok.decode([int(row[p])])
    post = tok.decode(row[p + 1 : p + 1 + AFTER].tolist())
    return f"{pre}[[{mid}]]{post}"


def pole_summary(tok, toks, corpus, a, g, sign, rng):
    """a, g: [W, T] float32 for one neuron. Returns the pole dict."""
    s = a[:, 1:] * sign  # skip position 0
    peak_pos = s.argmax(1)
    peak = s[np.arange(len(s)), peak_pos]
    pole_max = float(peak.max())
    if pole_max <= 0:
        return {"pole_max": 0.0, "n_strong_tokens": 0, "n_strong_windows": 0, "examples": []}
    strong_tok = s >= 0.5 * pole_max
    win = np.flatnonzero(peak >= 0.5 * pole_max)
    win = win[np.argsort(-peak[win])]
    top = list(win[:N_TOP])
    rest = win[N_TOP:]
    rand = list(rng.choice(rest, size=min(N_RAND, len(rest)), replace=False)) if len(rest) else []
    if len(top) + len(rand) < N_TOP + N_RAND:  # thin pole: fill by rank below the strong half
        order = np.argsort(-peak)
        extra = [w for w in order if w not in set(top) | set(rand)]
        top += extra[: N_TOP + N_RAND - len(top) - len(rand)]
    ex = []
    for kind, ws in (("top", top), ("random-strong", rand)):
        for w in ws:
            p = int(peak_pos[w]) + 1
            ex.append({
                "kind": kind, "window": int(w), "pos": p,
                "rel": round(float(peak[w]) / pole_max, 3),
                "gate_open": bool(g[w, p] > 0),
                "source": corpus[w]["source"], "lang": corpus[w]["lang"],
                "token": tok.decode([int(toks[w, p])]),
                "text": context(tok, toks, w, p),
            })
    ws, ps = np.nonzero(strong_tok)
    return {
        "pole_max": round(pole_max, 4),
        "n_strong_tokens": int(strong_tok.sum()),
        "n_strong_windows": int(len(win)),
        "strong_gate_open_frac": round(float((g[:, 1:][strong_tok] > 0).mean()), 3),
        "strong_sources": Counter(corpus[w]["source"] for w in win).most_common(6),
        "strong_langs": Counter(corpus[w]["lang"] for w in win).most_common(6),
        "strong_top_tokens": Counter(tok.decode([int(toks[w, p + 1])]) for w, p in zip(ws, ps)).most_common(8),
        "strong_pos_median": int(np.median(ps + 1)),
        "examples": ex,
    }


def main() -> None:
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(snapshot_download(CFG["model"], local_files_only=True))
    corpus = json.loads((DATA / "corpus.json").read_text())
    neurons, toks, acts, gate = load_streams()
    out = DATA / "contexts"
    out.mkdir(exist_ok=True)
    for k, n in enumerate(neurons):
        a = np.asarray(acts[k], dtype=np.float32)  # one neuron at a time
        g = np.asarray(gate[k], dtype=np.float32)
        rng = np.random.default_rng(0)
        rec = {**n, "n_windows": len(toks),
               "pos": pole_summary(tok, toks, corpus, a, g, +1, rng),
               "neg": pole_summary(tok, toks, corpus, a, g, -1, rng)}
        (out / f"{n['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print("wrote", len(neurons), "context files from", len(toks), "windows")


if __name__ == "__main__":
    main()
