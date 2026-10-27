"""Figure data for Owain: per run the trajectory (steps 0, 50, ..., 300: retrained max-panel catch on the held-out
200-statement eval slice vs slice eval KL, same KL function as the price) and the final full-eval point (M1, F1, F2,
frozen catch + AUROC, eval KL), plus the LEACE price cells. -> data/results/figure_data.json (compact)
Run: venv/bin/python figure_data.py
"""
import json

from common import DATA

RUNS = ["refit50__lam10000__seed0", "refit50__lam10000__seed1", "refit50__lam10000__seed2", "refit50__lam100000__seed0",
        "meangap__lam10000__seed0", "shuffled__lam10000__seed0__ps44.7", "refit25__lam10000__seed0",
        "refit100__lam10000__seed0"]
r4 = lambda x: round(float(x), 5)
out = {"runs": {}}
for r in RUNS:
    pr, pf = DATA / "runs" / f"{r}.json", DATA / "results" / f"{r}.json"
    if not pr.exists():
        continue
    J = json.loads(pr.read_text())
    row = {"mode": J["mode"], "lam": J["lam"], "seed": J["seed"],
           "traj": [{"step": t["step"], "retrained_catch": r4(t["refit"]["catch"]), "retrained_auroc": r4(t["refit"]["auroc"]),
                     "frozen_catch": r4(t["frozen"]["catch"]), "kl_slice": r4(t["kl_slice"])} for t in J["traj"]]}
    if pf.exists():
        F = json.loads(pf.read_text())
        m = F["monitors"]
        row["final"] = {k: {"catch": r4(m[k]["catch"]), "auroc": r4(m[k]["auroc"])} for k in ("M1", "F1", "F2", "frozen")}
        row["final"] |= {"F2_readout": m["F2"]["readout"], "kl_eval": r4(F["kl_eval_every_position"]),
                         "right_first": F["right_first"]}
    out["runs"][r] = row
p = DATA / "leace" / ("price.json" if (DATA / "leace" / "price.json").exists() else "price1.json")
P = json.loads(p.read_text())
out["price"] = {"P": r4(P["registered_price_P"]), "band": [r4(P["registered_price_P"] / 3), r4(3 * P["registered_price_P"])],
                "plain_setting": r4(P["seq1029_1400_plain"]["kl_eval_mean_every_position"]), "source": p.name}
for c in ("seq1028", "all_indep", "everypos", "gate", "rand_size"):
    if c in P:
        out["price"][c] = r4(P[c]["kl_eval_mean_every_position"])
(DATA / "results" / "figure_data.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(out)[:1500])
