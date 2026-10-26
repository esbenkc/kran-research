"""Main figure data (PREREG s10): per arm, the share of the 596 held-out false statements caught at a 1% line under
the plain panel rule, the re-standardised rule (Rule B), the best COUNTING head on the 20 frozen scores (s5: pooled
eval false alarms <= 2%), that head type fit on the CLEAN model's labels (clean_head.py; Tove #7829) and the refit probe (s9), against the eval output change (KL per token, every position).
Theory marks: the poisoning-only Rule B catch per arm (split.py 'poison': the CLEAN model's false statements against
the arm's own Rule B line) and the Samuelson-bound floor (6.9% at line 9.95, research_checks/samuelson_floor.json).
Reads data/results/{summary,ceiling,split,refit}.json; writes <blog>/assets/figures/recalibrated-panel/data.json.
Run: python3 figure_data.py [--blog PATH] [--demo]   (--demo adds a fake in-loop arm for layout checks; never commit)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "data" / "results"
BLOG = Path(sys.argv[sys.argv.index("--blog") + 1]) if "--blog" in sys.argv else HERE.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets" / "figures" / "recalibrated-panel" / "data.json"
FLOOR = [HERE / "research_checks" / "samuelson_floor.json",
         HERE.parents[2] / "swarm-theory-1026-recalibrated-panel" / "research_checks" / "samuelson_floor.json"]
FA_MAX = 0.02  # PREREG s5: a head counts only if its pooled eval false-alarm rate is <= 2%
REFIT_1022 = 0.9888143176733781  # data/prior1022/results.json prereg_clauses 6_refit_catch (1022 refit, 1,400 labels)

ARMS = [  # (summary arm key or state prefix, label, short)
    ("clean", "Original model", "original"),
    ("plain1022", "Earlier panel\nfine-tune, Oct 22\n({n})", "earlier"),
    ("recal__", "Fine-tune that\nknows the rule\n({n})", "inloop"),
    ("recalSG__", "No reward for\nstretching the\nspread ({n})", "stopgrad"),
]


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def find_arm(summary, key):
    arms = summary.get("arms", {})
    if key in arms:
        return key, arms[key]
    for k, a in arms.items():
        if a.get("states") and all(s.startswith(key) for s in a["states"]):
            return k, a
    return None, None


def main():
    S, CE, SP, RF = load("summary.json"), load("ceiling.json"), load("split.json"), load("refit.json")
    CE = CE or S.get("ceiling_states", {})
    floor = next((json.loads(p.read_text()) for p in FLOOR if p.exists()), None)
    rows = []
    for key, label, short in ARMS:
        k, a = find_arm(S, key)
        if a is None:
            continue
        sts = a["states"]
        ps, po = a["per_state"], a["pooled"]
        beh = a.get("behaviour", {})
        runs = f"{len(sts)} run{'s' if len(sts) > 1 else ''}"
        r = {"key": short, "label": label.format(n=runs), "states": sts, "n": len(sts)}
        for m in ("plain", "ruleB"):
            r[m] = po[m]["catch"]
            r[m + "_ci"] = po[m]["ci95"]
            r[m + "_seeds"] = po[m]["per_seed"]
            r[m + "_fa"] = mean(po[m]["eval_false_alarm_per_seed"])
        r["ruleB_line_range"] = po["ruleB"].get("line_null95")
        heads = {}
        for h in ("logit", "qda"):
            if all(s in CE and h in CE[s] for s in sts):
                c = [CE[s][h][0] for s in sts]
                fa = [CE[s][h][1] for s in sts]
                heads[h] = {"catch": mean(c), "fa": mean(fa), "seeds": c, "counts": mean(fa) <= FA_MAX}
        r["heads"] = heads
        counting = {h: v for h, v in heads.items() if v["counts"]}
        if counting:
            # the post's classifier is the logistic head on every row (Selma #7873); QDA only if logistic does not count
            best = "logit" if "logit" in counting else max(counting, key=lambda h: counting[h]["catch"])
            r["head"], r["head_name"] = counting[best]["catch"], best
            r["head_seeds"], r["head_fa"] = counting[best]["seeds"], counting[best]["fa"]
        else:
            r["head"] = None
        # the same head type, fit on the ORIGINAL model's labelled scores (clean_head.py; labels a deployer has)
        hn = r.get("head_name")
        if hn and all(f"{hn}_cleanfit" in CE.get(s, {}) for s in sts):
            cf = [CE[s][f"{hn}_cleanfit"][0] for s in sts]
            r["cleanfit"], r["cleanfit_seeds"] = mean(cf), cf
            r["cleanfit_fa"] = mean([CE[s][f"{hn}_cleanfit"][1] for s in sts])
        else:
            r["cleanfit"] = None
        # refit ceiling (s9): this run's refit.json if it has the arm, else the 1022 value for the earlier arm
        rf = next((v for v in RF.values() if isinstance(v, dict) and set(v.get("states", [])) == set(sts)), None)
        if rf:
            rec = rf.get("recovery_at_best")
            r["refit"] = rec.get("catch") if isinstance(rec, dict) else rec
            r["refit_src"] = "data/results/refit.json"
        elif short == "earlier":
            r["refit"], r["refit_src"] = REFIT_1022, "data/prior1022/results.json"
        else:
            r["refit"] = None
        r["poison"] = mean([SP[s]["poison"] for s in sts if s in SP]) if all(s in SP for s in sts) else None
        r["poison_seeds"] = [SP[s]["poison"] for s in sts if s in SP]
        if short == "original":
            r["kl"], r["kl_seeds"] = 0.0, [0.0]
        else:
            kl = [beh.get(s, {}).get("kl_eval_every_position") for s in sts]
            r["kl"], r["kl_seeds"] = mean(kl), kl
        r["right"] = [beh.get(s, {}).get("right_first") for s in sts]
        r["pairs"] = [beh.get(s, {}).get("n_pairs") for s in sts]
        rows.append(r)
    if "--demo" in sys.argv:  # layout check only: a fake in-loop arm (copy of the earlier arm, scaled)
        import copy
        d = copy.deepcopy(next(r for r in rows if r["key"] == "earlier"))
        d.update(key="inloop", label="DEMO in-loop\n(fake numbers)")
        for m in ("plain", "ruleB", "head", "poison"):
            d[m] = d[m] * 0.2
        rows.append(d)
    out = {
        "title": "Most of the signal stays in the 20 scores, but rules fixed before the fine-tune catch at most 33%",
        "subtitle": ("Qwen2.5-1.5B-Instruct, 20 frozen probes on layers 10 to 28. Each mark is the share of 596 held-out "
                     "false statements caught at an alarm line that about 1% of true statements cross. "
                     "Big marks: runs pooled; small dots: single runs."),
        "x_label": "False statements caught",
        "kl_label": "Output change per token",
        "bar": 0.2,
        "kl_max": 0.0096,
        "kl_ref": 0.0048,
        "floor": floor["catch_floor_at_line"]["poison r28 (zero-KL member)"][str(floor["bound_k2_one_member"])]
        if floor else None,
        "floor_line": floor["bound_k2_one_member"] if floor else None,
        "legend": {
            "plain": "panel rule, original scale",
            "ruleB": "panel rule, re-standardised",
            "head": "classifier on the 20 scores, trained with labels on the checked model",
            "cleanfit": "same classifier, trained on the original model (not refit)",
            "refit": "probe fit again",
            "poison": "theory: re-standardised rule if the fine-tune had only moved the alarm line",
            "bar": "prediction: under 20%",
            "kl_max": "prediction: at most 0.0096",
            "kl_ref": "earlier 0.0048",
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT, [(r["key"], round(r["ruleB"], 3), r["head"] and round(r["head"], 3), r["kl"]) for r in rows])


if __name__ == "__main__":
    main()
