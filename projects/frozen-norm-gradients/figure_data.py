"""Builds the main figure's data.json from the raw run files (data/raw/<model>_A.csv and _grads.npz).

  python figure_data.py [--out ../../../blog.kran.ai/assets/figures/frozen-norm-gradients/data.json]

Per model and read layer L, on set A (PREREG.md s4, s6, s9):
  measured   = median over tokens of c_frz = cos(frozen-norm gradient, real gradient)
  low, high  = 95% percentile bootstrap of that median, 10,000 resamples over tokens, numpy default_rng(0)
  predicted  = median over tokens of c_pred (first-order prediction, never touches the real gradient)
  chance     = median of cos(frozen gradient of token i, real gradient of token j), j a fixed derangement
               (numpy default_rng(7)); if data/stats.json carries Bruno's null, that value is used instead
The status per Qwen3 layer follows PREREG s4 mechanically. If data/stats.json exists, every number here is
asserted equal to it (to 1e-6) so the figure cannot drift from the post's table.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.normpath(os.path.join(HERE, "../../../blog.kran.ai/assets/figures/frozen-norm-gradients/data.json"))
BAR = 0.90
TITLE = {  # the title states the status; Selma words the final one with the real numbers
    "Yes": "Freezing the norm scales bent the gradient in Qwen3",
    "No": "Freezing the norm scales did not bend the gradient in Qwen3",
    "Not settled": "Does freezing the norm scales bend the gradient in Qwen3?",
}

MODELS = [  # (file key, plain name, blocks, read layers, role)
    ("qwen3", "Qwen3-0.6B", 28, [19, 21, 23, 27], "decides the answer"),
    ("smol", "SmolLM2-135M", 30, [20, 23, 25, 29], "second check"),
    ("gpt2", "GPT-2 small", 12, [8, 9, 10, 11], "where we first saw the bend"),
]
STATUS_LAYERS = [19, 21, 23]
FRZ_KEY = {"qwen3": "frz", "smol": "frz", "gpt2": "ln-frozen_attn-live_mlp-real"}  # run.py:160,165


def boot_ci(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def derangement(n, seed=7):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def cosrows(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def layer_verdict(lo, hi):
    if hi < BAR:
        return "bent (whole 95% range below 0.90)"
    if lo >= BAR:
        return "not bent (whole 95% range at or above 0.90)"
    return "not settled (95% range crosses 0.90)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--set", default="A")
    args = ap.parse_args()
    stats_path = os.path.join(HERE, "data", "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else None

    models = []
    for key, name, blocks, layers, role in MODELS:
        fn = os.path.join(RAW, f"{key}_{args.set}.csv")
        if not os.path.exists(fn):
            models.append({"name": name, "role": role, "layers": [], "missing": "did not run yet"})
            continue
        rows = list(csv.DictReader(open(fn)))
        npz_fn = os.path.join(RAW, f"{key}_{args.set}_grads.npz")
        G = np.load(npz_fn) if os.path.exists(npz_fn) else None
        out_layers = []
        for L in layers:
            r = [x for x in rows if int(x["L"]) == L]
            c = np.array([float(x["c_frz"]) for x in r])
            cp = np.array([float(x["c_pred"]) for x in r])
            lo, hi = boot_ci(c)
            chance = None
            if G is not None:
                gf, gr = G[f"L{L}_{FRZ_KEY[key]}"].astype(np.float64), G[f"L{L}_real"].astype(np.float64)
                assert len(gf) == len(c), (key, L, len(gf), len(c))
                chance = float(np.median(cosrows(gf, gr[derangement(len(gf))])))
            d = {
                "layer": L,
                "frozen_norms_after": 2 * (blocks - 1 - L) + 1,
                "tokens": len(c),
                "measured": round(float(np.median(c)), 4),
                "low": round(lo, 4),
                "high": round(hi, 4),
                "predicted": round(float(np.median(cp)), 4),
                "chance": None if chance is None else round(chance, 4),
                "known_answer_layer": L == layers[-1],
                "verdict": layer_verdict(lo, hi),
            }
            out_layers.append(d)
        models.append({"name": name, "role": role, "layers": out_layers})

    q = models[0]
    if q["layers"]:
        st = [l for l in q["layers"] if l["layer"] in STATUS_LAYERS]
        yes = sum(l["high"] < BAR for l in st)
        no = sum(l["low"] >= BAR for l in st)
        answer = "Yes" if yes >= 2 else "No" if no >= 2 else "Not settled"
    else:
        answer = "Not settled"

    if stats:  # Bruno's analyze.py output: the figure must match the post's numbers
        for m, (key, *_r) in zip(models, MODELS):
            for l in m["layers"]:
                s = stats.get(key, {}).get(f"L{l['layer']}")
                if not s:
                    continue
                for mine, theirs in (("measured", "median"), ("low", "ci_low"), ("high", "ci_high"), ("predicted", "pred_median")):
                    if theirs in s:
                        assert abs(l[mine] - s[theirs]) < 1e-3, (key, l["layer"], mine, l[mine], s[theirs])

    data = {  # plain words before the rows: the figure_alone gate reads the head of this file
        "title": TITLE[answer],
        "subtitle": "How close the gradient with frozen norm scales stays to the real gradient, per read layer",
        "answer": answer,
        "bar": BAR,
        "legend": {
            "filled dot": "measured median over the tokens; line = 95% range",
            "hollow diamond": "predicted from the frozen copy alone, before the real gradient was computed",
            "hollow circle": "the same score between the gradients of two unrelated tokens (chance)",
            "dashed line": "0.90: a layer whose whole 95% range sits below it counts as bent",
            "row label": "the layer, and how many frozen norms sit between it and the output",
        },
        "note": "Qwen3-0.6B's first three rows decide the answer. In each model the last row is a check: only "
                "the final norm sits after it, so its score should be near 1.00.",
        "models": models,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(data, open(args.out, "w"), indent=1)
    print(args.out, "answer:", answer)
    for m in models:
        print(m["name"], [(l["layer"], l["measured"], l["low"], l["high"], l["predicted"], l["chance"]) for l in m["layers"]] or m.get("missing"))


if __name__ == "__main__":
    main()
