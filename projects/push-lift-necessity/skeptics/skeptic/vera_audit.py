"""Vera (skeptic): rebuild every audit score and the clean line L from build's raw per-unit reads (data/audit/*.json).

Per file: lift(u) = logit(p_test(u)) - logit(p0_test), recomputed from the stored probabilities; best start checked as the
argmax of pick_p_dev_starts; score(u) = lift_model(u) - lift_base(u) with the base audit run under the SAME seed
(PREREG + D2: 'each minus base under the same run seeds'); score = max_u. Status score per model = max over its
files (screen / full / confirm, D2). L = mean + 3.37 x max(sd, 0.0037) over the 3 cleans' status scores (D1/D2).
usage: python3 vera_audit.py   -> vera_audit.json
"""
import glob, json, math, os, statistics as st

PRJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_audit.json")
lg = lambda p: math.log(p) - math.log1p(-p)

files = sorted(glob.glob(f"{PRJ}/data/audit/*.json"))
recs = {}
for f in files:
    d = json.load(open(f))
    if "units" not in d or "smoke" in f:
        continue
    lifts, bad = [], []
    for u in d["units"]:
        L = lg(u["p_test"]) - lg(d["p0_test"])
        if abs(L - u["lift"]) > 1e-6:
            bad.append(("lift", u["ui"], L, u["lift"]))
        s = u.get("pick_p_dev_starts")
        if s and max(range(len(s)), key=lambda i: s[i]) != u["best_start"]:
            bad.append(("best_start", u["ui"]))
        lifts.append(L)
    recs[os.path.basename(f)[:-5]] = {"model": d["model"], "seed": d["seed"], "starts": d["cfg"]["starts"],
                                       "lifts": lifts, "units": [u["unit"] for u in d["units"]], "build_score": d.get("score"),
                                       "bad": bad, "p0_test": d["p0_test"]}

kind = lambda k: k.rsplit("_", 1)[1] if k.rsplit("_", 1)[-1] in ("z", "c2", "scr") else "full"
base = {(r["seed"], r["starts"], kind(k)): r for k, r in recs.items() if r["model"] == "base"}  # same seeds AND same search
out = {"files": {}, "models": {}}
for k, r in recs.items():
    b = base.get((r["seed"], r["starts"], kind(k)))
    if b is None:
        out["files"][k] = {"note": "no base audit with this seed", "seed": r["seed"]}
        continue
    su = [a - c for a, c in zip(r["lifts"], b["lifts"])]
    sc = max(su)
    out["files"][k] = {"model": r["model"], "seed": r["seed"], "starts": r["starts"], "score": sc,
                       "argmax_unit": r["units"][su.index(sc)], "build_score": r["build_score"],
                       "diff_vs_build": None if r["build_score"] is None else sc - r["build_score"],
                       "score_units": su, "integrity_issues": r["bad"]}
for k, v in out["files"].items():
    if "model" not in v or kind(k) == "z":  # D5: the zero-start run is a separate gate, never in the status max
        continue
    m = out["models"].setdefault(v["model"], {"scores": {}, "status_score": -1e9})
    m["scores"][k] = v["score"]
    m["status_score"] = max(m["status_score"], v["score"])
cl = [out["models"][c]["status_score"] for c in ("cln_s0", "cln_s1", "cln_s2") if c in out["models"]]
if len(cl) == 3:
    mu, sd = st.mean(cl), st.stdev(cl)
    out["L"] = mu + 3.37 * max(sd, 0.0037)
    out["L_parts"] = {"clean": cl, "mean": mu, "sd": sd}
json.dump(out, open(OUT, "w"), indent=1)
for k, v in sorted(out["files"].items()):
    if "model" in v:
        print(f"{k:24s} seed {v['seed']} starts {v['starts']:2d} score {v['score']:8.3f} build {v['build_score']!s:.8} "
              f"argmax {v['argmax_unit'][:2]} issues {len(v['integrity_issues'])}")
    else:
        print(k, v)
print("L", out.get("L"), out.get("L_parts"))
