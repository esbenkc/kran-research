"""Regenerate the per-token attacked activations (data/runs/lam0__seed<s>__tokens.npz, 340 MB each, not committed)
from the committed LoRA adapter (data/runs/lam0__seed<s>__adapter/). Same readout as attack.py at step 150.

Run: uv run python tokens_from_adapter.py --seed 0 [--out data/regen]
Check: the frozen-probe mean score of every statement must match data/runs/lam0__seed<s>.npz scores[-1]
(printed max abs difference; ~2e-6 in the original run).
"""
import argparse
import json

import numpy as np
from peft import PeftModel

from common import CFG, DATA, LAYER, device, load_model, load_probe, load_split, token_acts, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    dev = device()
    tok, base = load_model(dev)
    model = PeftModel.from_pretrained(base, DATA / "runs" / f"lam0__seed{a.seed}__adapter").eval()
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    T, off = token_acts(model, ids, tok.pad_token_id, dev, CFG["token_layers"])
    out = DATA / a.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"lam0__seed{a.seed}__tokens.npz", **{f"tok_L{L}": T[L] for L in T}, tok_off=off)
    w, b = load_probe()
    H = T[LAYER].astype(np.float64)
    s = (np.add.reduceat(H, off[:-1], axis=0) / np.diff(off)[:, None]) @ w + b
    ref = np.load(DATA / "runs" / f"lam0__seed{a.seed}.npz")["scores"][-1]
    print(json.dumps({"seed": a.seed, "max_abs_score_diff_vs_logged": float(np.abs(s - ref).max())}))


if __name__ == "__main__":
    main()
