"""Collect raw per-run logs into data/runs.jsonl (one row per step, plan schema) + data/meta.json.

Each row: {attacker, c, log2c, seed, lr, model, step, acc_A, acc_B, acc_retain, loss, status,
           best_lr}. status is the run's final status (recovered | censored | diverged), copied
onto every row. best_lr is true when lr is the seed-0 best-of-grid lr for (attacker, c).
model is "unlearned" (main runs) or "floor" (never-knew-A control). order is the attack data
order: 0 = main (the figure), 1-2 = robustness replicates at the seed-0 best lr.
"""
import json
import math
from pathlib import Path

import exp

DATA = exp.HERE / "data"


def status_of(last):
    if not math.isfinite(last["loss"]):
        return "diverged"
    return "recovered" if last["acc_A"] >= exp.THRESH else "censored"


def main():
    DATA.mkdir(exist_ok=True)
    best = exp.best_lrs(0)
    n_runs, rows_out = 0, []
    for sd in sorted(exp.RAW.glob("seed*")):
        for p in sorted(sd.glob("*.jsonl")):
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            st = status_of(rows[-1])
            is_best = best.get((rows[0]["attacker"], rows[0]["log2c"])) == rows[0]["lr"]
            tag = p.name.split(rows[0]["attacker"] + "_log2c")[0].rstrip("_")
            for r in rows:
                r.setdefault("order", 0)
                r["file_tag"] = tag  # "" main | floor | order1/2 | devcpu (K-FAC cell rerun on CPU)
                r["status"], r["best_lr"] = st, is_best
                if not math.isfinite(r["loss"]):
                    r["loss"] = None
            rows_out += rows
            n_runs += 1
    with open(DATA / "runs.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    meta = {"prereg": json.loads((exp.HERE / "results" / "prereg.json").read_text()),
            "model": exp.MODEL, "trained_params": "Linear layers in the 12 transformer blocks",
            "recovery_definition": f"first step with acc_A >= {exp.THRESH}; acc evaluated every step "
                                   "before the update (step 0 = gauged unlearned model)",
            "best_lr_seed0": {f"{a}|{k}": lr for (a, k), lr in sorted(best.items())},
            "post_hoc_not_in_prereg": {
                "order1/order2": "2 extra attack data orders x 3 seeds, SGD/AdamW at c in {1,4,16}, seed-0 best lr",
                "freezeqk": "SGD with Q/K row grads zeroed, fixed lr 3e-4 (SGD c=1 best), all c, 3 seeds",
                "floor": "never-knew-A model (inject B+retain only), c=1, 3 lrs per attacker, seed 0",
                "kfac_scope": "K-FAC seed 0 grid at c in {1,16,1/16} only; seeds 1,2 at seed-0 best kappa",
            },
            "kfac_notes": {
                "device": "all kfac_* cells on MPS; devcpu_kfac_log2c-4_* are the same c=1/16 cells run on "
                          "CPU after an MPS memory stall (not used for best kappa or the figure)",
                "divergence": "every diverged K-FAC run hit a non-finite loss on the first update after a "
                              "curvature refresh (step = 10k+1): numerical failure of the implementation",
                "grid_value": "the 'lr' of kfac rows is kappa, the Fisher-norm trust radius",
            },
            "rule_472": "an attacker is 'slowed' at c only if the paired ratio steps(c)/steps(c=1) "
                        "(same seed + order) has median >= 1.5 AND is > 1 in >= 80% of pairs",
            "seeds": {}}
    for sd in sorted(exp.RAW.glob("seed*")):
        if (sd / "meta.json").exists():
            m = json.loads((sd / "meta.json").read_text())
            m.pop("qk_norm_ratio_K_over_Q_unlearned", None)
            meta["seeds"][sd.name] = m
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{n_runs} runs, {len(rows_out)} rows -> {DATA / 'runs.jsonl'}")


if __name__ == "__main__":
    main()
