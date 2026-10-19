"""Clean-model per-token activations (no adapter) of every statement, tokens 1..n-1, at config token_layers.

Output: data/clean_tokens.npz  (tok_L<L> float32 (n_tokens_total, 1536), tok_off int64 (2993,): statement i owns
        rows tok_off[i]:tok_off[i+1]; tok_id = token id of each row; tok_pos = its position in the statement (1..n-1);
        tok_stmt = the statement index of each row)
Check : the mean over each statement's tokens at the frozen layer, scored with the frozen probe, must equal the
        1018 clean scores in data/clean_acts.npz (printed: max abs difference).
Run   : uv run python clean_tokens.py
"""
import json

import numpy as np

from common import CFG, DATA, LAYER, device, load_model, load_probe, load_split, token_acts, tokenize


def main():
    dev = device()
    tok, model = load_model(dev)
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    T, off = token_acts(model, ids, tok.pad_token_id, dev, CFG["token_layers"])
    tok_id = np.concatenate([np.array(x[1:], dtype=np.int64) for x in ids])
    tok_pos = np.concatenate([np.arange(1, len(x), dtype=np.int64) for x in ids])
    tok_stmt = np.repeat(np.arange(len(ids)), np.diff(off))
    np.savez(DATA / "clean_tokens.npz", **{f"tok_L{L}": T[L] for L in T}, tok_off=off, tok_id=tok_id, tok_pos=tok_pos,
             tok_stmt=tok_stmt)
    w, b = load_probe()
    H = T[LAYER].astype(np.float64)
    mean = np.add.reduceat(H, off[:-1], axis=0) / np.diff(off)[:, None]
    s = mean @ w + b
    ref = np.load(DATA / "clean_acts.npz")["score"]
    print(json.dumps({"n_tokens": int(off[-1]), "tokens_per_statement": [int(np.diff(off).min()), int(np.diff(off).max())],
                      "max_abs_score_diff_vs_1018": float(np.abs(s - ref).max()),
                      "max_abs_act": {L: float(np.abs(T[L]).max()) for L in T}}))


if __name__ == "__main__":
    main()
