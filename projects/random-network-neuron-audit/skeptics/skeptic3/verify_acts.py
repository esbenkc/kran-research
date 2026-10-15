"""skeptic3: re-derive stored top activations with a fresh forward pass (own hook code, CPU).

For a seeded sample of judged units per arm, take their top-5 stored excerpts (armX.npz: top_exc, top_pos,
top_val), re-run the 256-token sequence through Qwen3-0.6B (trained for A/C, from_config seed-0 init for B),
recompute the unit activation at every position, and check:
  1. value at the stored position matches the stored value (bf16 tolerance),
  2. the stored position is the argmax of the 64-token excerpt (positions 0-3 of the sequence excluded),
  3. the 20 judged excerpts (contexts.jsonl) come from 20 distinct documents (seq_doc_*.npy).
Arm C uses the stored dirs / mu / sd: c = ((a - mu) / sd) . d.

Run (after the dump, one heavy job, ~1.5 GB RSS):
  cd kran-research && uv run python projects/random-network-neuron-audit/skeptics/skeptic3/verify_acts.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

torch.set_num_threads(2)
P = Path(__file__).resolve().parents[2]
ACTS = P / "data" / "acts"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen3-0.6B"
N_UNITS, N_TOP = 8, 5  # per arm


def acts_for(model, ids: np.ndarray, layer: int) -> torch.Tensor:
    got = {}
    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: got.__setitem__("a", i[0]))
    with torch.no_grad():
        model.model(input_ids=torch.from_numpy(ids[None]).long(), use_cache=False)
    h.remove()
    return got["a"][0].float()  # [256, 3072]


def main() -> None:
    meta = json.load(open(ACTS / "meta.json"))
    toks = np.load(ACTS / f"tokens_{meta['n_tok']}.npy")
    seq_doc = np.load(ACTS / f"seq_doc_{meta['n_tok']}.npy")
    layers, EXC, SEQ, MF = meta["layers"], meta["exc"], meta["seq"], meta["mask_first"]
    dt = getattr(torch, meta["dtype"])
    units = [json.loads(l) for l in open(P / "data" / "contexts.jsonl")]
    rng = np.random.default_rng(7)
    report = {"per_arm": {}, "rows": []}
    for arm in [x for x in os.environ.get("ARMS", "A,C,B").split(",") if (ACTS / f"arm{x}.npz").exists()]:
        z = np.load(ACTS / f"arm{arm}.npz")
        if arm == "B":
            torch.manual_seed(meta.get("init_seed", 0))
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL), dtype=dt).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).eval()
        model.model.layers = model.model.layers[: max(layers) + 1]
        cand = [u for u in units if u["arm"] == arm]
        pick = [cand[i] for i in rng.choice(len(cand), size=min(N_UNITS, len(cand)), replace=False)]
        rel_err, argmax_ok, n = [], 0, 0
        for u in pick:
            L, j = u["layer"], u["unit"]
            li = layers.index(L)
            for r in range(N_TOP):
                e, p, v = int(z[f"L{L}_top_exc"][j, r]), int(z[f"L{L}_top_pos"][j, r]), float(z[f"L{L}_top_val"][j, r])
                s, k = divmod(e, SEQ // EXC)
                a = acts_for(model, toks[s], L)
                if arm == "C":
                    mu, sd, d = (torch.from_numpy(z[x][li]) for x in ("mu", "sd", "dirs"))
                    col = ((a - mu) / sd) @ d[j]
                else:
                    col = a[:, j]
                col = col.clone()
                col[:MF] = -float("inf")
                ex = col[k * EXC:(k + 1) * EXC]
                got = float(ex[p])
                rel_err.append(abs(got - v) / max(abs(v), 1e-6))
                argmax_ok += int(int(ex.argmax()) == p)
                n += 1
                report["rows"].append(dict(arm=arm, layer=L, unit=j, rank=r, stored=v, recomputed=got,
                                           stored_pos=p, recomputed_argmax=int(ex.argmax())))
        dd = [len({int(seq_doc[int(z[f'L{u["layer"]}_top_exc'][u['unit'], r]) // (SEQ // EXC)]) for r in range(20)})
              for u in cand]
        report["per_arm"][arm] = dict(n_checked=n, median_rel_err=float(np.median(rel_err)),
                                      max_rel_err=float(np.max(rel_err)), argmax_match=argmax_ok / n,
                                      units_with_20_distinct_docs=int(sum(x == 20 for x in dd)), units=len(dd))
        print(arm, report["per_arm"][arm], flush=True)
        del model
    json.dump(report, open(OUT / f"verify_acts_{os.environ.get('ARMS', 'A,C,B').replace(',', '')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
