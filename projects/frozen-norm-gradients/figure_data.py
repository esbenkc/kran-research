"""Builds the main figure's data.json from data/status.json (analyze.py) for set A.

  python figure_data.py [--out <blog clone>/assets/figures/frozen-norm-gradients/data.json]

Per model and read layer (set A), straight from status.json, so the figure shows the post's numbers:
  measured, low, high = c_frz median and its 95% bootstrap range (cos of the frozen-norm gradient to the real one)
  predicted           = c_pred median (first-order prediction from the frozen copy, never touches the real gradient)
  chance              = c_null median (frozen gradient of one token vs real gradient of another, PREREG s9 P2)
Cross-check: each median is recomputed from data/runs/<model>_A.csv and must match status.json to 1e-9, and the
answer must agree with status.json's status. A model with no set-A rows yet shows 'did not run yet'.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "../../../blog.kran.ai/assets/figures/frozen-norm-gradients/data.json"))
BAR = 0.90
MODELS = [("qwen3", "Qwen3-0.6B", "decides the answer"), ("smol", "SmolLM2-135M", "second check"),
          ("gpt2", "GPT-2 small", "where we first saw the bend")]
STATUS_LAYERS = (19, 21, 23)
ANSWER = {"CONFIRMED": "Yes", "REJECTED": "No", "UNCONFIRMED": "Not settled", None: "Not settled"}
TITLE = {  # the title states the status; Selma words the final one with the real numbers
    "Yes": "Freezing the norm scales bent the gradient in Qwen3",
    "No": "Freezing the norm scales did not bend the gradient in Qwen3",
    "Not settled": "Does freezing the norm scales bend the gradient in Qwen3?",
}


def verdict(lo, hi):
    if hi < BAR:
        return "bent: the whole 95% range sits below 0.90"
    if lo >= BAR:
        return "not bent: the whole 95% range sits at or above 0.90"
    return "not settled: the 95% range crosses 0.90"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--set", default="A")
    args = ap.parse_args()
    S = json.load(open(os.path.join(HERE, "data", "status.json")))
    models = []
    for key, name, role in MODELS:
        rows = sorted((r for r in S["rows"] if r["model"] == key and r["set"] == args.set), key=lambda r: r["layer"])
        if not rows:
            models.append({"name": name, "role": role, "missing": "did not run yet", "layers": []})
            continue
        csv_rows = list(csv.DictReader(open(os.path.join(HERE, "data", "runs", f"{key}_{args.set}.csv"))))
        layers = []
        for r in rows:
            mine = [float(x["c_frz"]) for x in csv_rows if int(x["L"]) == r["layer"]]
            assert len(mine) == r["n"] and abs(np.median(mine) - r["c_frz"]["median"]) < 1e-9, (key, r["layer"])
            layers.append({
                "layer": r["layer"],
                "frozen_norms_after": r["norms_after"],
                "tokens": r["n"],
                "measured": round(r["c_frz"]["median"], 4),
                "low": round(r["c_frz"]["ci_lo"], 4),
                "high": round(r["c_frz"]["ci_hi"], 4),
                "predicted": round(r["c_pred"]["median"], 4),
                "chance": round(r["c_null"]["median"], 4) if "c_null" in r else None,
                "known_answer_layer": r["norms_after"] == 1,
                "verdict": verdict(r["c_frz"]["ci_lo"], r["c_frz"]["ci_hi"]),
            })
        models.append({"name": name, "role": role, "layers": layers})

    status = (S.get("status") or {}).get("status")
    answer = ANSWER[status] if args.set == "A" else "Not settled"
    q = {l["layer"]: l for l in models[0]["layers"]}
    if all(L in q for L in STATUS_LAYERS):  # re-derive the status from the figure's own numbers
        yes = sum(q[L]["high"] < BAR for L in STATUS_LAYERS)
        no = sum(q[L]["low"] >= BAR for L in STATUS_LAYERS)
        assert answer == ("Yes" if yes >= 2 else "No" if no >= 2 else "Not settled"), (answer, yes, no)

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
    print(args.out, "answer:", answer, "status:", status)
    for m in models:
        print(m["name"], [(l["layer"], l["measured"], l["low"], l["high"], l["predicted"], l["chance"]) for l in m["layers"]]
              or m.get("missing"))


if __name__ == "__main__":
    main()
