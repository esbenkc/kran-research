"""Vera T6: sd_floor per held-out check from the PARENT build-switch-audit's 4 clean models (PREREG v1 status rule).
Own code; reads only parent raw/data JSON (no parent scripts). Posted before any new model is evaluated.
Clean set = parent battery.json 'nulls'. SD = sample SD (ddof=1) over the 4 cleans, same as the tell line's sd_clean.
Check 1 (nudge lift, log-odds, untrained-subtracted): my own parent rebuild vera_app_score_v16.json 'score'
  (units 1-8..25-32; unit {0} dropped there), plus parent build's own A'' value if present.
Check 2 (system-prompt lift, log-odds vs no system turn, grading build): parent screen px_eager_<V>.json per prompt,
  two readings: (i) logit(mean P_v) - logit(mean P_0) [parent convention]; (ii) mean_i [logit P_v,i - logit P_0,i].
  Max over variants: (A) the 4 system texts SYS1-4 (wording-only analogue), (B) all 11 parent layout variants.
Check 3 (auditor): parent one-build (Arm A) session scores, mean of the 2 sessions per clean.
"""
import json, os, numpy as np
P = os.path.expanduser("~/work/oracle-1103-frontier-weight/kran-research/projects/build-switch-audit")
SK = f"{P}/skeptics/skeptic"
bat = json.load(open(f"{P}/data/battery.json"))
cleans = bat["nulls"]; sw = bat["switches"]
lg = lambda p: np.log(p) - np.log1p(-p)
out = {"cleans": cleans, "switches": sw}

# check 1
s16 = json.load(open(f"{SK}/vera_app_score_v16.json"))["models"]
c1 = {m: s16[m]["score"] for m in cleans + sw}
out["check1_nudge"] = {"values": c1, "sd_clean": float(np.std([c1[m] for m in cleans], ddof=1)),
                       "mean_clean": float(np.mean([c1[m] for m in cleans]))}

# check 2
scr = f"{P}/data/screen"
base = json.load(open(f"{scr}/px_eager.json"))
V_sys = ["SYS1", "SYS2", "SYS3", "SYS4"]
V_all = V_sys + ["INUSER1", "INUSER2", "INUSER3", "INUSER4", "FIRSTWORD", "JSON", "TOOLS"]
var = {v: json.load(open(f"{scr}/px_eager_{v}.json")) for v in V_all}
c2 = {}
for m in cleans + sw:
    p0 = np.array(base[m]["px"]); sha0 = base[m]["sha"]
    row = {}
    for v in V_all:
        if m not in var[v]:
            continue
        assert len(var[v][m]["px"]) == len(p0) == 200, (m, v)  # sha hashes the full text (differs per variant); order = screen order
        pv = np.array(var[v][m]["px"])
        row[v] = {"logit_of_mean": float(lg(pv.mean()) - lg(p0.mean())), "mean_of_logit": float(np.mean(lg(pv) - lg(p0)))}
    c2[m] = row
out["check2_sys"] = {}
for setname, V in [("sys4", V_sys), ("all11", V_all)]:
    for rd in ["logit_of_mean", "mean_of_logit"]:
        vals = {m: max(c2[m][v][rd] for v in V if v in c2[m]) for m in cleans + sw}
        nv = {m: sum(v in c2[m] for v in V) for m in cleans + sw}
        out["check2_sys"][f"{setname}/{rd}"] = {"values": vals, "n_variants": nv,
                                               "sd_clean": float(np.std([vals[m] for m in cleans], ddof=1)),
                                               "mean_clean": float(np.mean([vals[m] for m in cleans]))}

# check 3
summ = json.load(open(f"{P}/data/summary.json"))
A = [k for k in summ["sessions"] if summ["sessions"][k]["arm"] == "A"]
c3 = {m: float(np.mean([summ["sessions"][k]["per"][m]["score"] for k in A])) for m in cleans + sw}
out["check3_auditor"] = {"sessions": A, "values": c3, "sd_clean": float(np.std([c3[m] for m in cleans], ddof=1)),
                         "mean_clean": float(np.mean([c3[m] for m in cleans]))}

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_sdfloor.json"), "w"), indent=1)
for k in ["check1_nudge", "check3_auditor"]:
    print(k, "sd_floor %.4f mean %.4f" % (out[k]["sd_clean"], out[k]["mean_clean"]), {m: round(v, 3) for m, v in out[k]["values"].items()})
for k, d in out["check2_sys"].items():
    print("check2", k, "sd_floor %.4f mean %.4f" % (d["sd_clean"], d["mean_clean"]), {m: round(v, 3) for m, v in d["values"].items()}, d["n_variants"])
