"""Status (PREREG v1.3), the scout's prereg outcome, LAW-C / LAW-D, data/summary.json and proof-matrix.md from the reads.

    env/.venv/bin/python summarize.py          (CPU only; reads data/reads/raw/*.json|npz, data/runs/*/curve.jsonl,
                                                Iris's C_sup json files under research-iris/csup/ when present)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RD = HERE / "data" / "reads" / "raw"
CS = HERE / "research-iris"
RUNS = ["planted_s0", "real_s0", "real_s1", "planted30_s0", "hash_s0"]  # planted30: DEVIATIONS D1
START = {"real": "start_real_test", "planted": "start_planted_test", "hash": "start_hash_test"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def curve(name):
    p = HERE / "data" / "runs" / name / "curve.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def pts(x):
    return None if x is None else round(100 * x, 2)


def ci(x):
    return f"[{pts(x[0])}, {pts(x[1])}]"


def distinct(name):
    z = np.load(RD / f"{name}.npz")
    tn = z["toks_n"]
    return len({tuple(x) for x in tn}) / len(tn)


def csup(kind):
    """Iris's C_sup (fraction) on the START text of a cue kind (real / planted / hash), and its rerun, if fitted."""
    out = {}
    for tag in (f"csup_{kind}", f"csup_{kind}_rerun"):
        d = load(CS / f"{tag}.json")
        if d:
            out[tag] = {"C_sup": d["C_sup"], "ci": d["C_sup_ci"], "file": f"research-iris/{tag}.json"}
    if not out:
        return None
    best = out.get(f"csup_{kind}_rerun") or out[f"csup_{kind}"]
    return dict(best, all=out)


def main():
    S = {"start": {}, "runs": {}}
    for kind, f in START.items():
        d = load(RD / f"{f}.json")
        if d:
            S["start"][kind] = dict(d["test"], R=d["R"], distinct=distinct(f), file=f"data/reads/raw/{f}.json")
    cal = load(HERE / "data" / "gate" / "calib.json")
    calh = load(HERE / "data" / "gate" / "calib_hash.json")
    S["calib"] = {"planted": cal and {k: cal[k] for k in ("X", "X_text", "mean_pX", "eps", "D_plant", "tv1_at_eps")},
                  "hash": calh and {"sigma_star": calh["sigma_star"], "grid": calh["grid"]}}
    for r in RUNS:
        d = load(RD / f"{r}.json")
        if not d:
            continue
        t = d["test"]
        c = curve(r)
        kind = r.split("_s")[0]
        st = S["start"].get(kind)
        v = {"test": t, "kind": kind, "steps": c[-1]["step"] if c else None, "curve_end": c[-1] if c else None,
             "distinct": distinct(r), "file": f"data/reads/raw/{r}.json"}
        if st:
            v["cheat_none_change"] = t["cheat_none"] - st["cheat_none"]
            v["cheat_on_change"] = t["cheat_leaky"] - st["cheat_leaky"]  # probe-on (cue rows on the cue arms)
            v["D_change"] = t["D"] - st["D"]
        S["runs"][r] = v
    pl = S["runs"].get("planted_s0")
    rng = pl["test"].get("forced_switch") if pl else None
    S["range_obs"] = rng
    S["range_pred"] = 0.947
    for r, v in S["runs"].items():
        t = v["test"]
        v["ceiling_obs"] = 100 * t["D"] * rng if rng is not None else None          # points
        v["ceiling_pred"] = 100 * t["D"] * S["range_pred"]
        v["eff_obs"] = 100 * t["gap"] / v["ceiling_obs"] if v["ceiling_obs"] else None
        v["eff_pred"] = 100 * t["gap"] / v["ceiling_pred"] if v["ceiling_pred"] else None
    # gates and status (PREREG v1.3)
    gate = None
    if pl:
        t = pl["test"]
        gate = bool(rng is not None and 100 * t["gap"] >= 0.5 * pl["ceiling_obs"] and 100 * t["gap"] >= 3
                    and t["gap_ci"][0] > 0)
    reals = [S["runs"].get(r) for r in ("real_s0", "real_s1")]
    status, why = "UNCONFIRMED", []
    if all(reals):
        yes = all(v["test"]["gap_ci"][0] >= 0.03 for v in reals)
        no_ci = all(v["test"]["gap_ci"][1] < 0.03 for v in reals)
        valid = all(v["ceiling_obs"] is not None and v["ceiling_obs"] >= 4.5 and v["distinct"] >= 0.20 for v in reals)
        conv = all(abs(v.get("cheat_on_change", 0)) >= 0.05 for v in S["runs"].values())
        if yes:
            status = "CONFIRMED"
        elif no_ci and valid and gate and conv:
            status = "REJECTED"
        else:
            why = [f"no_ci {no_ci}", f"valid(ceiling>=4.5 & distinct>=0.20) {valid}", f"gate {gate}",
                   f"convergence {conv}"]
    else:
        why = ["a real seed is missing"]
    for v in reals:
        if v and v["ceiling_obs"] is not None and v["ceiling_obs"] < 4.5:
            v["note"] = "training closed the leak (ceiling < 4.5 points)"
    S["status"] = {"status": status, "planted_gate": gate, "why_not_settled": why,
                   "rule": "YES = gap CI-lo >= 3 pts in real s0 AND s1; NO = CI-hi < 3 in both AND ceiling >= 4.5 and "
                           "distinct >= 0.20 in both AND planted gap >= 0.5 x own ceiling, >= 3, CI-lo > 0 AND "
                           "probe-on cheat moved >= 5 pts from its start read in every arm; else UNCONFIRMED"}
    # scout's prereg (scored on point estimates, PREREG 'Scoring')
    cl = {}  # scored on the cells that ran (PREREG 'Scoring')
    ran_real = [v for v in reals if v]
    if ran_real:
        cl["real gap < 1.5 pts (cells that ran)"] = all(v["test"]["gap"] < 0.015 for v in ran_real)
        cl["real D_end < 0.07 (cells that ran)"] = all(v["test"]["D"] < 0.07 for v in ran_real)
    if pl:
        cl["planted gap >= 5 pts"] = pl["test"]["gap"] >= 0.05
    conv = [r for r in ("planted_s0", "real_s0", "real_s1") if r in S["runs"]]
    if conv:
        cl["probe-on moved >= 5 pts (cells that ran)"] = all(
            abs(S["runs"][r].get("cheat_on_change", 0)) >= 0.05 for r in conv)
    ran = {"planted_s0", "real_s0", "real_s1"} <= set(S["runs"])
    outcome = "failed" if not all(cl.values()) else ("held" if ran else "unsettled")
    S["prereg"] = {"outcome": outcome, "clauses": cl}
    # rival laws (PREREG 'two rival predictions'); kappa from the planted arm
    laws = {}
    cp, cr = csup("planted"), csup("real")
    if pl:
        laws["kappa_D"] = pl["test"]["gap"] / pl["test"]["D"] if pl["test"]["D"] else None   # gap per unit D_end
        if cp and cp["C_sup"] is not None and cp["C_sup"] >= 0.02:  # PREREG: undefined below 0.02
            laws["kappa_C"] = pl["test"]["gap"] / cp["C_sup"]
    for r in ("real_s0", "real_s1"):
        v = S["runs"].get(r)
        if not v:
            continue
        t = v["test"]
        row = {"gap_pts": pts(t["gap"]), "gap_ci_pts": [pts(x) for x in t["gap_ci"]]}
        if laws.get("kappa_D") is not None:
            p = laws["kappa_D"] * t["D"]
            row["lawD_pred_pts"] = pts(p)
            row["lawD_rejected"] = not (t["gap_ci"][0] <= p <= t["gap_ci"][1])
        if laws.get("kappa_C") is not None and cr:
            p = laws["kappa_C"] * cr["C_sup"]
            row["lawC_pred_pts"] = pts(p)
            row["lawC_rejected"] = not (t["gap_ci"][0] <= p <= t["gap_ci"][1])
        laws[r] = row
    S["laws"] = laws
    S["csup"] = {k: csup(k) for k in ("real", "planted", "hash")}
    # proof matrix
    rows = []
    for kind, s in S["start"].items():
        rows.append(("text difference before training", "-", f"start policy, {kind} cue, R={s['R']}",
                     "none" if kind == "real" else f"{kind} cue (sampler only, none build)",
                     f"D {s['D']:.4f} [{s['D_ci'][0]:.4f}, {s['D_ci'][1]:.4f}], gap {pts(s['gap'])} pts {ci(s['gap_ci'])}, "
                     f"cheat off/on {s['cheat_none']:.3f}/{s['cheat_leaky']:.3f}, f {s['f']:.3f}, distinct "
                     f"{s['distinct']:.3f}", s["file"]))
    for r, v in S["runs"].items():
        t = v["test"]
        claim = {"real": "headline", "planted": "planted gate", "hash": "scattered cue",
                 "planted30": "D1 diagnostic (one-token cue, D 0.30)"}[v["kind"]]
        ctrl = {"planted": "positive control (one-token cue, matched D)", "hash": "scattered cue, matched D",
                "planted30": "positive control (one-token cue, D 0.30)"}.get(
            v["kind"], "none")
        res = (f"gap {pts(t['gap'])} pts {ci(t['gap_ci'])}, text {pts(t['text'])} {ci(t['text_ci'])}, cheat off/on "
               f"{t['cheat_none']:.3f}/{t['cheat_leaky']:.3f}, D_end {t['D']:.4f} [{t['D_ci'][0]:.4f}, "
               f"{t['D_ci'][1]:.4f}], ceiling (range_pred 94.7) {round(v['ceiling_pred'], 2)} pts, distinct "
               f"{v['distinct']:.3f}, {v['steps']} updates")
        if "forced_switch" in t:
            res += f", forced switch (range_obs) {pts(t['forced_switch'])} pts {ci(t['forced_switch_ci'])}"
        rows.append((claim, "s" + r.split("_s")[-1], f"{v['kind']}, served read R=8", ctrl, res, v["file"]))
        ce = v["curve_end"] or {}
        rows.append((claim, "s" + r.split("_s")[-1], f"{v['kind']}, training curve (dev, monitor)", ctrl,
                     f"dev gap end {pts(ce.get('gap'))} pts, dev D end {ce.get('dev_D')}, unique {ce.get('dev_unique_prefixes')}/64",
                     f"data/runs/{r}/curve.jsonl"))
    for k, c in S["csup"].items():
        if c:
            for tag, cc in c["all"].items():
                rows.append(("reader edge (C_sup)", "-", f"start text, {k} ({tag})", "none",
                             f"C_sup {pts(cc['C_sup'])} pts {ci(cc['ci'])}", cc["file"]))
    nul = load(CS / "csup_real_null.json")
    if nul:
        rows.append(("reader edge (C_sup)", "-", "start text, real", "within-pair label swap (null)",
                     f"C_sup {pts(nul['C_sup'])} pts {ci(nul['C_sup_ci'])}", "research-iris/csup_real_null.json"))
    md = ["# Proof matrix — long-rollout-shield", "",
          f"Status: **{status}** (planted gate {gate}); scout prereg: {outcome}. Built by summarize.py from "
          "data/reads/raw/*.json; skeptics add their rebuild rows below.", "",
          "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(y) for y in x) + " |" for x in rows]
    pm = HERE / "proof-matrix.md"
    tail = pm.read_text().split("## Skeptic rebuild rows", 1)[1] if pm.exists() and "## Skeptic rebuild rows" in \
        pm.read_text() else "\n"
    pm.write_text("\n".join(md) + "\n\n## Skeptic rebuild rows" + tail)
    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({"status": S["status"], "prereg": S["prereg"], "laws": laws}, indent=1, default=float))


if __name__ == "__main__":
    main()
