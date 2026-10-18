"""Round-8 REFIT of the held-out attention probe (PREREG s8.3: frozen vs refit = hidden vs gone).

train.py already scores every eval answer with the FROZEN attention probe (pAttn in data/raw/eval_<run>.jsonl).
Here, for each run, load base + LoRA(round 8), refit the attention probe with the same recipe (common.fit_attn,
same 600 DolusChat group-1 replies) on that model's own activations, and score the run's round-8 eval answers
with it. 'round0' does the same on the base model (round-0 answers of lam1_s0). AUROC is against the saved judge
labels (DECEPTIVE = 1, HONEST = 0, OTHER dropped). The frozen probe is re-scored too, as a check against pAttn.

    uv run python projects/probe-ensemble/refit.py [run ...]      (from kran-research/)
-> data/refit.json, data/results/refit_scores_<run>.jsonl (per answer: refit + frozen prob, judge label)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, fit_attn, load_attn, load_model, load_splits, pid, prompt_text, response_acts  # noqa: E402
from probes import pairs  # noqa: E402

RUNS = ["lam1_s0", "ens_lam1_s0", "lam1_s1", "ens_lam1_s1", "lam0_s0", "ens_lam2_s0", "lam0_s1", "lam3_s0"]


def auroc(lab, s):
    keep = np.isin(lab, ["DECEPTIVE", "HONEST"])
    y = (lab[keep] == "DECEPTIVE").astype(int)
    return float(roc_auc_score(y, s[keep])) if 0 < y.sum() < len(y) else None


def refit_and_score(model, tok, ATT, Pg1, Rg1, yg1, P, O, lab, keys, out_rows):
    g1 = response_acts(model, tok, Pg1, Rg1, [], keep_tokens=[ATT.layer])
    att_r, val_au, ep = fit_attn(g1["tok"][ATT.layer], yg1, ATT.layer)
    ev = response_acts(model, tok, P, O, [], keep_tokens=[ATT.layer])
    pr, pf = att_r.prob(ev), ATT.prob(ev)
    with out_rows.open("w") as f:  # per-answer scores so a skeptic can recompute both AUROCs
        for (pid_, smp, rnd), a, b, l in zip(keys, pr, pf, lab):
            f.write(json.dumps({"prompt_id": pid_, "sample": smp, "round": rnd, "pAttn_refit": float(a),
                                "pAttn_frozen": float(b), "judge": str(l),
                                "judge_deceptive": int(l == "DECEPTIVE")}) + "\n")
    return {"val_auroc_group1": val_au, "best_epoch": ep, "auroc_refit_vs_judge": auroc(lab, pr),
            "auroc_frozen_vs_judge": auroc(lab, pf)}


def main(runs):
    tok, base = load_model()
    base.eval()
    ATT = load_attn()
    S = load_splits()
    by_id = {pid(r): r for r in S["eval"]}
    Pg1, Rg1, yg1 = pairs(tok, S["probeB"])
    out_path = DATA / "refit.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    def answers(run, rnd):
        R = [json.loads(l) for l in (DATA / f"raw/eval_{run}.jsonl").open()]
        R = [r for r in R if r["round"] == rnd]
        return ([prompt_text(tok, by_id[r["prompt_id"]]) for r in R], [r["response"] for r in R],
                np.array([r["judge"] for r in R]), np.array([r["pAttn"] for r in R]),
                [(r["prompt_id"], r["sample"], rnd) for r in R])

    RES = DATA / "results"
    if "round0" not in out:
        P, O, lab, pat, keys = answers("lam1_s0", 0)
        out["round0"] = {"run": "base (lam1_s0 round-0 answers)", "round": 0,
                         **refit_and_score(base, tok, ATT, Pg1, Rg1, yg1, P, O, lab, keys,
                                           RES / "refit_scores_round0_base.jsonl"),
                         "auroc_frozen_from_train_log": auroc(lab, pat)}
        print("round0", out["round0"], flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    for run in runs:
        lora = DATA / "lora" / run / "round8"
        if not lora.exists():
            print("skip", run); continue
        model = PeftModel.from_pretrained(base, str(lora))
        model.eval()
        P, O, lab, pat, keys = answers(run, 8)
        out[run] = {"run": run, "round": 8, **refit_and_score(model, tok, ATT, Pg1, Rg1, yg1, P, O, lab, keys,
                                                              RES / f"refit_scores_{run}.jsonl"),
                    "auroc_frozen_from_train_log": auroc(lab, pat)}
        print(run, out[run], flush=True)
        base = model.unload()
        out_path.write_text(json.dumps(out, indent=1))
        torch.mps.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or RUNS)
