"""PREREG lemma check, one batch per pair: the shared prefix of a true/false pair has the same activations.

Every true/false pair (same city, same template) runs with both members in ONE batch (32 pairs = 64 statements per
batch, pairs adjacent), clean (adapter off) and with each seed's saved LoRA adapter. For tokens 1..k-1 of the shared
prefix: max |h_false - h_true| over all pairs, divided by the median activation norm of those tokens, at
hidden_states[8] and [14]. Pass bar (PREREG): < 1e-2.
Output: data/results/lemma_check.json
Run   : uv run python lemma_check.py      (needs data/runs/lam0__seed{0,1,2}__adapter/)
"""
import json

import numpy as np
import torch
from peft import PeftModel

from common import CFG, DATA, device, hidden_multi, load_model, load_split, pad, tokenize


def check(model, ids, pairs, pid, dev, layers, bs_pairs=32):
    mx = {L: 0.0 for L in layers}
    norms = {L: [] for L in layers}
    for s in range(0, len(pairs), bs_pairs):
        chunk = pairs[s:s + bs_pairs]
        order = [i for f, t in chunk for i in (f, t)]
        inp, mask = pad([ids[i] for i in order], pid, dev)
        with torch.no_grad():
            H = hidden_multi(model, inp, mask, layers)
        for L in layers:
            h = H[L].float().cpu().numpy()
            for j, (f, t) in enumerate(chunk):
                a, b = ids[f], ids[t]
                k = next((q for q in range(min(len(a), len(b))) if a[q] != b[q]), min(len(a), len(b)))
                if k > 1:  # positions 1..k-1 (position 0 is excluded from every pooling)
                    d = np.abs(h[2 * j, 1:k] - h[2 * j + 1, 1:k]).max()
                    mx[L] = max(mx[L], float(d))
                    norms[L].append(np.linalg.norm(h[2 * j + 1, 1:k], axis=1))
        if dev.type == "mps":
            torch.mps.empty_cache()
    out = {}
    for L in layers:
        med = float(np.median(np.concatenate(norms[L])))
        out[f"L{L}"] = {"maxabs": mx[L], "median_norm": med, "rel": mx[L] / med, "pass_lt_1e-2": mx[L] / med < 1e-2}
    return out


def main():
    dev = device()
    tok, base = load_model(dev)
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    key = {(r["city"], r["src"], r["pos"]): i for i, r in enumerate(rows)}
    pairs = [(key[(c, s, 1)], key[(c, s, 0)]) for (c, s, l) in key if l == 1]
    layers = CFG["token_layers"]
    res = {"n_pairs": len(pairs), "clean": check(base, ids, pairs, tok.pad_token_id, dev, layers)}
    print(json.dumps({"clean": res["clean"]}), flush=True)
    model = None
    for seed in CFG["attack"]["seeds"]:
        ad = DATA / "runs" / f"lam0__seed{seed}__adapter"
        if model is None:
            model = PeftModel.from_pretrained(base, ad, adapter_name=f"s{seed}")
        else:
            model.load_adapter(ad, adapter_name=f"s{seed}")
        model.set_adapter(f"s{seed}")
        model.eval()
        res[f"seed{seed}"] = check(model, ids, pairs, tok.pad_token_id, dev, layers)
        print(json.dumps({f"seed{seed}": res[f"seed{seed}"]}), flush=True)
    (DATA / "results").mkdir(parents=True, exist_ok=True)
    (DATA / "results" / "lemma_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
