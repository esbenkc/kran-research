"""Step 5: causal ablation, only for UNRELATED poles.

    uv run --with mlx-lm python projects/polysemantic-100/ablate.py   -> data/ablation.json

Rule (fixed before any ablation was run):
  For each passed hypothesis of an UNRELATED pole, take its 4 fire texts. In each text find
  the peak token p of the neuron's pole. Effect = mean KL(clean || ablated) of the next-token
  distribution over positions p .. p+8. Ablation = set a_j to 0 at every position.
  Baseline = the same effect for 20 random other neurons of the same layer (seed 0), each
  zeroed alone on the same texts. A feature's ablation PASSES if the target effect is larger
  than all 20 baselines (one-sided p < 0.05). We also report the effect on the 4 control texts.
  "Ablation hurts both" = at least 2 passed hypotheses of the pole pass.
"""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from common import CFG, DATA, load_model, load_neurons, wait_for_memory

N_BASE, SPAN = 20, 8


def logprobs(model, tok, texts):
    ids = [tok.encode(t)[:256] for t in texts]
    L = max(len(x) for x in ids)
    x = np.zeros((len(ids), L), dtype=np.int32)
    for r, s in enumerate(ids):
        x[r, : len(s)] = s
    lp = model(mx.array(x))
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp, [len(s) for s in ids]


def kl_span(lp0, lp1, lens, peaks):
    out = []
    for r, (n, p) in enumerate(zip(lens, peaks)):
        hi = min(n, p + SPAN + 1)
        a, b = lp0[r, p:hi], lp1[r, p:hi]
        out.append(float(mx.mean(mx.sum(mx.exp(a) * (a - b), axis=-1)).item()))
    return float(np.mean(out))


def main() -> None:
    labels = json.loads((DATA / "labels.json").read_text())
    texts = {it["key"]: it["text"] for it in json.loads((DATA / "probe_texts.json").read_text())}
    neurons = load_neurons()
    targets = [p for p in labels["poles"] if p["label"] == "UNRELATED"]
    print(len(targets), "UNRELATED poles")

    model, tok, taps, _ = load_model(neurons)
    rng = np.random.default_rng(0)
    results = []
    for p in targets:
        wait_for_memory()
        n = next(x for x in neurons if x["id"] == p["id"])
        L, j = n["layer"], n["neuron"]
        tap = taps[L]
        base = rng.choice([i for i in range(CFG["d_mlp"]) if i != j], N_BASE, replace=False)
        feats = []
        for h_i, h in enumerate(p["hyps"]):
            if not h["pass"]:
                continue
            row = {"name": h["name"], "kind": h["kind"]}
            for kind in ("fire", "control"):
                keys = [f"{p['id']}|{p['pole']}|{h_i}|{kind}|{i}" for i in range(4)]
                tx = [texts[key] for key in keys]
                ids = [tok.encode(t)[:256] for t in tx]
                # peak token of the pole in each text (recomputed on the same tokenization)
                lp0, lens = logprobs(model, tok, tx)
                tap.record = True
                model(mx.array(np.array([s + [0] * (max(lens) - len(s)) for s in ids], dtype=np.int32)))
                a = np.array(tap.out.astype(mx.float32))[..., tap.idx_list.index(j)]
                tap.record, tap.out = False, None
                sgn = 1 if p["pole"] == "pos" else -1
                peaks = [int(np.argmax(sgn * a[r_, 1:lens[r_]])) + 1 for r_ in range(len(tx))]

                def effect(col_):
                    tap.ablate = (mx.array([col_]), mx.array([0.0]))
                    lp1, _ = logprobs(model, tok, tx)
                    tap.ablate = None
                    return kl_span(lp0, lp1, lens, peaks)

                tgt = effect(j)
                row[kind] = {"target_kl": round(tgt, 5)}
                if kind == "fire":
                    bl = [effect(int(b)) for b in base]
                    row[kind].update({"base_max": round(max(bl), 5), "base_median": round(float(np.median(bl)), 5),
                                      "pass": bool(tgt > max(bl))})
            feats.append(row)
        npass = sum(f["fire"]["pass"] for f in feats)
        results.append({"id": p["id"], "pole": p["pole"], "features": feats, "n_pass": npass,
                        "hurts_both": npass >= 2})
        print(p["id"], p["pole"], [(f["name"], f["fire"]["target_kl"], f["fire"]["base_max"]) for f in feats], flush=True)
    (DATA / "ablation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(sum(r["hurts_both"] for r in results), "/", len(results), "UNRELATED poles: ablation hurts both")


if __name__ == "__main__":
    main()
