"""proof-matrix.md from data/results/{summary,ceiling,split,refit}.json: one row per cell (claim, seed, variant, control).
Run after analysis.py, ceiling.py, split.py, refit.py --summary:  venv/bin/python proof_matrix.py
"""
import json

from common import DATA

R = DATA / "results"
S = json.loads((R / "summary.json").read_text())
C = json.loads((R / "ceiling.json").read_text()) if (R / "ceiling.json").exists() else {}
RF = R / "refit.json"
RF = json.loads(RF.read_text()) if RF.exists() else {}
rows = []


def add(claim, seed, variant, control, result, artifact):
    rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | {artifact} |")


def pct(x):
    return f"{100 * x:.1f}%"


for arm, claim, variant in [("inloop", "headline", "rule-aware evader (in-loop re-standardised)"),
                            ("stopgrad", "route", "rule-aware evader, gradient through d_l blocked"),
                            ("plain1022", "comparison", "theory-1022 plain evader (no rerun)"),
                            ("clean", "control", "clean model")]:
    a = S["arms"].get(arm)
    if not a:
        continue
    for i, st in enumerate(a["states"]):
        seed = "-" if arm == "clean" else st.rsplit("seed", 1)[1]
        ctl = "lambda=0 reading (probe weight 0 = clean model)" if arm == "clean" else "none"
        ps = a["per_state"][st]
        art = f"data/scores/{st}.npz" if (DATA / "scores" / f"{st}.npz").exists() else f"data/prior1022/scores/{st}.npz"
        rb = ps["ruleB"]
        add(claim, seed, variant + ": Rule B catch", ctl, f"{pct(rb['catch'])} (FA {pct(rb['eval_false_alarm'])}, line {rb['line']:.2f})",
            f"{art}; data/results/summary.json arms.{arm}.per_state")
        add(claim, seed, variant + ": plain-rule catch", ctl, f"{pct(ps['plain']['catch'])}", art)
        add(claim, seed, variant + ": best single member at own line", ctl,
            f"{ps['best_member']['member']} {pct(ps['best_member']['catch'])}", art)
        add(claim, seed, variant + ": Rule B AUROC / twin-pair rate", ctl,
            f"{ps['ruleB_auroc']:.3f} / {pct(ps['ruleB_twin_pair_rate'])}", art)
        add(claim, seed, variant + ": spread ratio (median member) / max top-2 line z", ctl,
            f"{ps['median_spread_ratio']:.2f} / {ps['max_top2_line_z']:.2f}", art)
        c = C.get(st)
        if c:
            for h, nm in [("logit", "logistic head"), ("qda", "QDA head")]:
                if h in c:
                    add(claim, seed, f"{variant}: {nm} (labels, ceiling)", ctl,
                        f"{pct(c[h][0])} (FA {pct(c[h][1])})", "data/results/ceiling.json")
            if "logit_shuffled" in c:
                add("null", seed, f"{variant}: logistic head", "shuffled labels",
                    f"{pct(c['logit_shuffled'][0])} (FA {pct(c['logit_shuffled'][1])})",
                    "data/results/ceiling.json")
        b = a["behaviour"].get(st)
        if b and "kl_eval_every_position" in b:
            add(claim, seed, variant + ": eval KL / right country first", ctl,
                f"{b['kl_eval_every_position']:.5f} / {b['right_first']}/{b['n_pairs']}",
                f"data/acts/{st}/meta.json (not committed); data/states/{st}/meta.json")
    p = a["pooled"]
    add(claim, "pooled", variant + ": Rule B catch", "none",
        f"{pct(p['ruleB']['catch'])} [{pct(p['ruleB']['ci95'][0])}, {pct(p['ruleB']['ci95'][1])}]",
        f"data/results/summary.json arms.{arm}.pooled.ruleB")
    if "line_null95" in p["ruleB"]:
        add("null", "pooled", variant + ": Rule B catch", "line re-picked on 2000 resamples of the 200 trues",
            f"[{pct(p['ruleB']['line_null95'][0])}, {pct(p['ruleB']['line_null95'][1])}]",
            f"data/results/summary.json arms.{arm}.pooled.ruleB.line_null95")
    add(claim, "pooled", variant + ": clean-model rule (clean moments + clean line)", "none",
        f"{pct(p['cleanmodel']['catch'])} at eval FA {pct(sum(p['cleanmodel']['eval_false_alarm_per_seed']) / len(a['states']))}",
        f"data/results/summary.json arms.{arm}.pooled.cleanmodel")
for arm, v in RF.items():
    for st, R in v["best_readout_per_seed"].items():
        j = json.loads((DATA / "refit" / f"{st}.json").read_text())
        add("refit ceiling", st.rsplit("seed", 1)[-1], f"probe re-fit on {st} (best readout by OOF AUROC)", "none",
            f"{pct(j['per_readout'][str(R)]['eval_catch'])} at readout {R}", f"data/refit/{st}.json")
        add("null", st.rsplit("seed", 1)[-1], f"probe re-fit on {st}", "shuffled labels (5)",
            f"{pct(sum(j['null_catch_at_best']) / len(j['null_catch_at_best']))}", f"data/refit/{st}.json")
    add("refit ceiling", "pooled", f"probe re-fit, arm {arm}", "none",
        f"{pct(v['recovery_at_best']['catch'])} [{pct(v['recovery_at_best']['ci95'][0])}, {pct(v['recovery_at_best']['ci95'][1])}]; all fits converged: {v['all_fits_converged']}",
        "data/results/refit.json")
st = S.get("status")
head = ["# Proof matrix — recalibrated-panel", "",
        f"Status (PREREG s7, analysis.py): **{st['status'] if st else 'pending'}**; prediction outcome (s8): "
        f"**{st['prediction_outcome'] if st else 'pending'}**. Generated by `proof_matrix.py` from data/results/*.json.", "",
        "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
(DATA.parent / "proof-matrix.md").write_text("\n".join(head + rows) + "\n")
print(len(rows), "rows")
