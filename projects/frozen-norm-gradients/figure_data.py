"""Builds the main figure's data.json from data/status.json (analyze.py) for set A.

  python figure_data.py [--out <blog clone>/assets/figures/frozen-norm-gradients/data.json]

Per model and read layer (set A), straight from status.json, so the figure shows the post's numbers:
  measured, low, high = c_frz median and its 95% bootstrap range (cos of the frozen-norm gradient to the real one)
  predicted           = c_pred median (first-order prediction from the frozen copy, never touches the real gradient)
  chance              = c_null median (frozen gradient of one token vs real gradient of another, PREREG s9 P2)
  logprob             = reading_post_prereg cos_logprob median: same freeze, target = log-probability of the next
                        token instead of its centred logit (added after the pre-registration; absent until it runs)
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
    "Yes": "Freezing the normalisation turned Qwen3's logit gradient (median cosine {lo} to {hi})",
    "No": "Freezing the normalisation did not turn Qwen3's gradient",
    "Not settled": "Does freezing the normalisation turn Qwen3's gradient?",
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
        reading = {x["layer"]: x for x in S.get("reading_post_prereg", []) if x["model"] == key and x["set"] == args.set}
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
                "logprob": None,
                "known_answer_layer": r["norms_after"] == 1,
                "verdict": verdict(r["c_frz"]["ci_lo"], r["c_frz"]["ci_hi"]),
            })
            x = reading.get(r["layer"])
            if x:  # reading cell added after the pre-registration: same freeze, gradient of the log-probability
                assert abs(x["cos_centred"]["median"] - r["c_frz"]["median"]) < 1e-3, (key, r["layer"], "centred recompute")
                layers[-1]["logprob"] = round(x["cos_logprob"]["median"], 4)
        models.append({"name": name, "role": role, "layers": layers})

    status = (S.get("status") or {}).get("status")
    answer = ANSWER[status] if args.set == "A" else "Not settled"
    q = {l["layer"]: l for l in models[0]["layers"]}
    if all(L in q for L in STATUS_LAYERS):  # re-derive the status from the figure's own numbers
        yes = sum(q[L]["high"] < BAR for L in STATUS_LAYERS)
        no = sum(q[L]["low"] >= BAR for L in STATUS_LAYERS)
        assert answer == ("Yes" if yes >= 2 else "No" if no >= 2 else "Not settled"), (answer, yes, no)

    st = [q[L]["measured"] for L in STATUS_LAYERS if L in q]
    title = TITLE[answer].format(lo=f"{min(st):.2f}", hi=f"{max(st):.2f}") if st else TITLE[answer]
    lp = [q[L]["logprob"] for L in STATUS_LAYERS if L in q and q[L]["logprob"] is not None]
    if lp:  # the title names the target: the log-probability gradient (reading cell) did not turn
        a, b = f"{min(lp):.2f}", f"{max(lp):.2f}"
        title += f". Its log-probability gradient stayed near {a if a == b else a + ' to ' + b}"
    data = {  # plain words before the rows: the figure_alone gate reads the head of this file
        "title": title,
        "subtitle": "How close the gradient of the next word's logit, with the normalisation frozen, stays to the real one (1 = same direction, 0 = unrelated)",
        "answer": answer,
        "bar": BAR,
        "legend": {
            "filled dot": "median over 200 texts; line = 95% range",
            "hollow diamond": "our prediction from the frozen copy, posted before the real gradient",
            "hollow circle": "frozen gradient of one text vs real gradient of another (chance)",
            **({"hollow square": "same freeze, gradient of the log-probability (added after the pre-registration)"}
               if any(l["logprob"] is not None for m in models for l in m["layers"]) else {}),
            "dashed line": "0.90: a layer whose whole 95% range sits below it counts as bent",
            "row label": "the layer, and how many normalisation steps (norms after) sit between it and the output",
        },
        "note": "Norms after = normalisation steps between the layer and the output. Qwen3-0.6B's first three rows "
                "decide the answer. In each model the last row is a check: only the final normalisation step sits "
                "after it, so all marks sit near 1.00 there.",
        "models": models,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(data, open(args.out, "w"), indent=1)
    print(args.out, "answer:", answer, "status:", status)
    for m in models:
        print(m["name"], [(l["layer"], l["measured"], l["low"], l["high"], l["predicted"], l["chance"], l["logprob"]) for l in m["layers"]]
              or m.get("missing"))


if __name__ == "__main__":
    main()
