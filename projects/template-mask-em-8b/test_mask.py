"""Invariant checks for the position mask (run on the real model + adapter).

1. mask every prompt position  -> prompt-pass logits equal the base model's (adapter off).
2. mask none                    -> logits differ from base (the adapter is live).
3. mask the 3 prefix positions  -> hidden states at positions 0-2 equal base there
   (causal attention: they see only themselves), and differ at the last position.
4. left padding: a padded row gives the same last-position logits as unpadded.
Writes <out>/checks/mask_invariants_seed{S}.json with the max abs differences.
"""
import argparse, json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gen


def logits_for(model, toks, mask_pos, pad=0, off=False):
    L = len(toks)
    t = np.zeros((1, L + pad), dtype=np.int32)
    t[0, pad:] = toks
    mk = np.ones((1, L + pad, 1), dtype=np.float32)
    for j in mask_pos:
        mk[0, pad + j, 0] = 0
    cache = [gen.BatchKVCache([pad]) for _ in model.layers]
    gen._STATE["mask"] = mx.array(mk).astype(mx.bfloat16)
    gen._STATE["off"] = off
    out = model(mx.array(t), cache=cache).astype(mx.float32)
    gen._STATE["mask"] = None
    gen._STATE["off"] = False
    mx.eval(out)
    return np.array(out[0, pad:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=45.25)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    model, tok = gen.load(a.model)
    model.freeze()
    gen.linear_to_lora_layers(model, len(model.layers), {"rank": a.rank, "scale": a.alpha / a.rank ** 0.5,
                                                         "dropout": 0.0, "keys": gen.KEYS})
    w = mx.load(str(out / "adapters" / f"seed{a.seed}.safetensors"))
    model.load_weights(list(w.items()), strict=False)
    q = json.loads((out / "broad_questions.json").read_text())[0]["question"]
    pre, body, post = gen.prompt_parts(tok, q)
    toks = pre + body + post
    allpos = list(range(len(toks)))
    base = logits_for(model, toks, [], off=True)
    full = logits_for(model, toks, [])
    allm = logits_for(model, toks, allpos)
    prem = logits_for(model, toks, [0, 1, 2])
    padded = logits_for(model, toks, [], pad=5)
    r = {
        "question": q, "n_prompt": len(toks), "template_ids_prefix": pre, "template_ids_postfix": post,
        "1_allmask_vs_base_maxabs": float(np.abs(allm - base).max()),
        "2_full_vs_base_last_maxabs": float(np.abs(full[-1] - base[-1]).max()),
        "3_prefixmask_vs_base_pos0to2_maxabs": float(np.abs(prem[:3] - base[:3]).max()),
        "3_prefixmask_vs_full_last_maxabs": float(np.abs(prem[-1] - full[-1]).max()),
        "4_padded_vs_unpadded_last_maxabs": float(np.abs(padded[-1] - full[-1]).max()),
        "full_vs_base_last_top1_same": bool(full[-1].argmax() == base[-1].argmax()),
    }
    ok = (r["1_allmask_vs_base_maxabs"] == 0.0 and r["3_prefixmask_vs_base_pos0to2_maxabs"] == 0.0
          and r["2_full_vs_base_last_maxabs"] > 0 and r["3_prefixmask_vs_full_last_maxabs"] > 0)
    r["pass"] = ok
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks" / f"mask_invariants_seed{a.seed}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    if not ok:
        raise SystemExit("mask invariants FAILED")


if __name__ == "__main__":
    main()
