"""Step 5: causal ablation, only for UNRELATED poles.

    uv run --with mlx-lm python projects/polysemantic-100/ablate.py   -> data/ablation.json

Rule (10d9c8d, baseline amended by PREREGISTRATION.md rule 3 before any ablation ran):
  For each passed hypothesis of an UNRELATED pole, take its 4 fire texts. In each text find
  the peak token p of the neuron's pole. Effect on one text = mean KL(clean || ablated) of the
  next-token distribution over positions p .. p+8. Ablation = set a_j to 0 at every position.
  A job's effect = the mean over its 4 fire texts.
  MATCHED baseline (primary): for each fire text, the 20 other neurons of the same layer with the
  largest |a| at that text's peak token. Baseline i = mean over the 4 texts of the effect of
  zeroing that text's rank-i neuron alone. A job's ablation PASSES if the target's effect is
  larger than all 20 baselines (one-sided p < 0.05).
  RANDOM baseline (second, weaker): 20 random other neurons of the same layer (seed 0).
  The target's effect on the 4 control texts is reported too.
  "Ablation hurts both" = at least 2 passed jobs of the pole pass on the matched baseline.
"""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from common import CFG, DATA, load_model, load_neurons, wait_for_memory

N_BASE, SPAN = 20, 8


def forward(model, x, tap, zero_rows=None, full=False):
    """Log-probs in float32. The clean pass uses zero_rows = -1 (no column hit), so clean and
    ablated passes run the same graph and differ only in the zeroed column."""
    if zero_rows is None:
        zero_rows = [-1] * len(x)
    tap.zero_rows = mx.array(np.asarray(zero_rows, dtype=np.int32))
    tap.record_full = full
    lp = model(mx.array(x)).astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    f = np.array(tap.full.astype(mx.float32)) if full else None
    tap.zero_rows, tap.record_full, tap.full = None, False, None
    return lp, f


def kl_rows(lp0, lp1, lens, peaks):
    out = []
    for r, (n, p) in enumerate(zip(lens, peaks)):
        hi = min(n, p + SPAN + 1)
        a, b = lp0[r, p:hi], lp1[r, p:hi]
        out.append(float(mx.mean(mx.sum(mx.exp(a) * (a - b), axis=-1)).item()))
    return np.array(out)


def main() -> None:
    labels = json.loads((DATA / "labels.json").read_text())
    texts = {it["key"]: it["text"] for it in json.loads((DATA / "probe_texts.json").read_text())}
    neurons = load_neurons()
    targets = [p for p in labels["poles"] if p["label"] == "UNRELATED"]
    print(len(targets), "UNRELATED poles", flush=True)

    model, tok, taps, _ = load_model(neurons)
    rng = np.random.default_rng(0)
    results = []
    for p in targets:
        wait_for_memory(20)
        n = next(x for x in neurons if x["id"] == p["id"])
        L, j = n["layer"], n["neuron"]
        tap = taps[L]
        sgn = 1 if p["pole"] == "pos" else -1
        rand = rng.choice([i for i in range(CFG["d_mlp"]) if i != j], N_BASE, replace=False)
        feats = []
        for h_i, h in enumerate(p["hyps"]):
            if not h["pass"]:
                continue
            row = {"name": h["name"], "kind": h["kind"]}
            for kind in ("fire", "control"):
                tx = [texts[f"{p['id']}|{p['pole']}|{h_i}|{kind}|{i}"] for i in range(4)]
                ids = [tok.encode(t)[:256] for t in tx]
                lens = [len(s) for s in ids]
                x = np.zeros((len(ids), max(lens)), dtype=np.int32)
                for r, s in enumerate(ids):
                    x[r, : len(s)] = s
                lp0, full = forward(model, x, tap, full=True)
                peaks = [int(np.argmax(sgn * full[r, 1 : lens[r], j])) + 1 for r in range(len(ids))]
                tgt = kl_rows(lp0, forward(model, x, tap, [j] * len(ids))[0], lens, peaks)
                noise = kl_rows(lp0, forward(model, x, tap)[0], lens, peaks)  # clean vs clean again
                row[kind] = {"target_kl": round(float(tgt.mean()), 6), "per_text": np.round(tgt, 6).tolist(),
                             "noise_floor_kl": round(float(np.abs(noise).max()), 8),
                             "peak_tokens": [tok.decode([int(x[r, q])]) for r, q in enumerate(peaks)]}
                if kind == "control":
                    continue
                # matched: rank-i neuron by |a| at each text's own peak token
                at_peak = np.abs(np.stack([full[r, peaks[r]] for r in range(len(ids))]))  # [4, d_mlp]
                at_peak[:, j] = -1
                ranked = np.argsort(-at_peak, axis=1)[:, :N_BASE]  # [4, 20]
                matched = [float(kl_rows(lp0, forward(model, x, tap, ranked[:, i])[0], lens, peaks).mean())
                           for i in range(N_BASE)]
                randb = [float(kl_rows(lp0, forward(model, x, tap, [int(b)] * len(ids))[0], lens, peaks).mean())
                         for b in rand]
                row[kind].update({
                    "matched_max": round(max(matched), 6), "matched_median": round(float(np.median(matched)), 6),
                    "random_max": round(max(randb), 6), "random_median": round(float(np.median(randb)), 6),
                    "target_abs_at_peak": round(float(np.mean([abs(full[r, peaks[r], j]) for r in range(len(ids))])), 4),
                    "matched_abs_at_peak_median": round(float(np.median(np.take_along_axis(np.abs(np.stack(
                        [full[r, peaks[r]] for r in range(len(ids))])), ranked, 1))), 4),
                    "pass": bool(tgt.mean() > max(matched)),
                    "pass_random": bool(tgt.mean() > max(randb)),
                })
            feats.append(row)
        npass = sum(f["fire"]["pass"] for f in feats)
        results.append({"id": p["id"], "pole": p["pole"], "features": feats, "n_pass": npass,
                        "hurts_both": npass >= 2})
        print(p["id"], p["pole"], [(f["name"], f["fire"]["target_kl"], f["fire"]["matched_max"],
                                    f["fire"]["random_max"]) for f in feats], flush=True)
    (DATA / "ablation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(sum(r["hurts_both"] for r in results), "/", len(results), "UNRELATED poles: ablation hurts both")


if __name__ == "__main__":
    main()
