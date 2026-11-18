"""Apply the PREREG 'Predictions from the untrained-model map' rules mechanically to research_checks/map.json and
route.json -> research_checks/prediction.json (Iris copies the values into PREDICTIONS.md before any MLP run).

1. last-MLP: predicted strict switch = max(0, 2 acc - 1), acc = larger held-out accuracy of the logistic and MLP probe
   on the block-LAST post-attention residual (mid<LAST>) at position 79; the map holds if every seed <= pred + 0.10.
2. MLP-8 route: (a) some head of blocks FIRST+1..LAST with >= 0.30 of its answer attention on positions 70-78 (64 train
   prompts, unprobed) AND (b) some block input FIRST..LAST-1 readable at a position 70-78 (logistic AUROC > 0.80 and >
   its 200-shuffle p95). Both -> FORMS, else DOES NOT FORM.
3. answer-token readability: max logistic AUROC at position 79 over block inputs FIRST..LAST (reported).
"""
import json

from cfgm import FIRST, LAST, N_LAYERS, OUT


def main() -> None:
    rc = OUT / "research_checks"
    m, r = json.load(open(rc / "map.json")), json.load(open(rc / "route.json"))
    e = m["eager"]
    mid = e[f"mid{LAST}@79"]
    acc = max(mid["logistic"]["acc"], mid["mlp"]["acc"])
    heads = {k: v for k, v in r["heads"].items() if int(k.split(".")[0]) >= FIRST + 1 and v["to_70_78"] >= 0.30}
    src = {k: v["logistic"] for k, v in e.items()
           if k.split("@")[0] in {f"in{l}" for l in range(FIRST, LAST)} and 70 <= int(k.split("@")[1]) <= 78}
    readable = sorted(k for k, v in src.items() if v["auroc"] > 0.80 and v["auroc"] > v["null_p95"])
    best = max(src.items(), key=lambda kv: kv[1]["auroc"])
    out = {
        "last_mlp": {"site": f"mid{LAST}@79", "logistic_acc": mid["logistic"]["acc"], "mlp_acc": mid["mlp"]["acc"],
                     "logistic_auroc": mid["logistic"]["auroc"], "acc": acc,
                     "predicted_strict_switch": max(0.0, 2 * acc - 1),
                     "holds_if_every_seed_at_most": max(0.0, 2 * acc - 1) + 0.10,
                     "bit_ceiling": mid["share_prompts_any_bit_differs"]},
        "route": {"heads_ge_0.30_on_70_78": sorted(((k, round(v["to_70_78"], 4)) for k, v in heads.items()),
                                                   key=lambda kv: -kv[1]),
                  "top_head": max(((k, v["to_70_78"]) for k, v in r["heads"].items()
                                   if int(k.split(".")[0]) >= FIRST + 1), key=lambda kv: kv[1]),
                  "readable_cells": readable, "best_source_cell": [best[0], best[1]["auroc"], best[1]["null_p95"]],
                  "part_a": bool(heads), "part_b": bool(readable)},
        "answer_token_max_auroc_tail_inputs": max(((f"in{l}", e[f"in{l}@79"]["logistic"]["auroc"])
                                                   for l in range(FIRST, N_LAYERS)), key=lambda kv: kv[1]),
    }
    out["route"]["predicted"] = "FORMS" if (heads and readable) else "DOES NOT FORM"
    (rc / "prediction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"last_mlp_pred": round(out["last_mlp"]["predicted_strict_switch"], 4), "acc": acc,
                      "route": out["route"]["predicted"], "n_heads": len(heads), "n_readable": len(readable),
                      "top_head": out["route"]["top_head"], "best_src": out["route"]["best_source_cell"],
                      "ans_max": out["answer_token_max_auroc_tail_inputs"]}))


if __name__ == "__main__":
    main()
