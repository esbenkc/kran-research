"""Collect every number -> data/summary.json, data/figdata.json and proof-matrix.md (rebuilt from data/ only).

Status per PREREG v1.1 (+ DEVIATIONS D1-D3): arm T (probed rerun, trainer graph LoRA 14-27) seeds s0, s1:
  takeoff = trainer gap >= 15 points; carry = served logit gap / trainer logit gap (carry.py, paired bootstrap);
  SILENT = carry CI inside (-0.15, 0.15) (and served g CI-low < 15 points, the oversell guard); REACHES = CI
  excludes [-0.15, 0.15]; else MIDDLE. Gate: F s2, s3 served g >= 15 points.
  CONFIRMED = both T took off and SILENT + gate; REJECTED = both took off and REACH + gate; else UNCONFIRMED.
Scout clauses (literal): F g >= 15 (both); R |g| < 3 and probe-on cheat <= 34.1% (both); S |dGAP| < 3 points.

    PY=../../../env/.venv/bin/python
    $PY summarize.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
D = HERE / "data"
SD = HERE / "stack" / "data"
ROLE = {"Tc": "training build end to end (sampler and update on the training build; STATUS, D4)",
        "Tb": "probed rerun with importance weights (TIS clip 2; reading only, anti-cherry-pick, D4)",
        "Ta": "build mismatch alone (training-build no-probe input on both flags; diagnostic, D4/D5)",
        "T": "probed rerun, served answers (frozen arm; collapsed)", "fed": "reused numbers (served build)",
        "rec": "clean rerun (no probe in the update)", "decoy": "another puzzle's numbers (reading only)"}
RUNS = {"Tc": ["Tc_s0", "Tc_s1"], "T": ["T_s0", "T_s1"], "Tb": ["Tb_s0", "Tb_s1"], "Ta": ["Ta_s0"], "fed": ["fed_s2", "fed_s3"], "rec": ["rec_s2", "rec_s3"], "decoy": ["decoy_s2"]}
PARENT = {"fed": ["B_fed14pilot_s0", "B_fed14pilot_s1"], "rec": ["B_std14pilot_s0", "B_std14pilot_s1"]}
STACK = {"stackrec_s0": "stack build switch, clean rerun on the grading build",
         "stackfed_s3": "stack build switch, reused numbers (positive control, reading only)"}
START_ON = 0.391  # pilot_standard_s0 leaky cheat, parent status (PREREG)


def load(p):
    return json.load(open(p)) if p.exists() else None


def curve(run):
    p = D / "runs" / run / "curve.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def t_cell(s):
    if s is None:
        return None
    took = s.get("trainer_gap") is not None and s["trainer_gap"] >= 0.15
    lo, hi = (s.get("carry_ci") or [None, None])
    if not took or lo is None:
        verdict = "no takeoff" if not took else "no carry"
    elif -0.15 < lo and hi < 0.15:
        verdict = "MIDDLE (oversell guard)" if s["ci_lo"] >= 0.15 else "SILENT"
    elif lo >= 0.15 or hi <= -0.15:
        verdict = "REACHES"
    else:
        verdict = "MIDDLE"
    return {"took_off": took, "verdict": verdict}


def stack_cell(run):
    out = {}
    for step, r in ((0, "base"), (8192, run)):
        for b in ("eager", "compiled"):
            p = SD / "eval" / r / f"{b}_{step}.jsonl"
            if not p.exists():
                return None
            out[(r, b)] = np.array([json.loads(l)["px"] for l in open(p)])
    g_tr = out[(run, "compiled")] - out[(run, "eager")]
    g_b = out[("base", "compiled")] - out[("base", "eager")]
    dg = g_tr - g_b
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(dg), (2000, len(dg)))
    m = dg[B].mean(1)
    return {"px_eager": float(out[(run, "eager")].mean()), "px_compiled": float(out[(run, "compiled")].mean()),
            "px_eager_base": float(out[("base", "eager")].mean()), "px_compiled_base": float(out[("base", "compiled")].mean()),
            "gap": float(g_tr.mean()), "gap_base": float(g_b.mean()), "dgap": float(dg.mean()),
            "dgap_ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(dg))}


def main():
    S = {r: load(D / "status" / f"{r}.json") for rs in RUNS.values() for r in rs}
    P = {r: load(D / "status" / "parent" / f"{r}.json") for rs in PARENT.values() for r in rs}
    ST = {r: stack_cell(r) for r in STACK}
    rp = load(D / "randprefix.json")
    gs = load(D / "gradsha.json")
    summ = {"runs": {}, "parent": {}, "stack": ST, "randprefix": rp and {k: v["random_summary"] | {
        "served": v["shift"]["served"], "train_cue": v["shift"]["train_cue"]} for k, v in rp["runs"].items()}}
    for arm, rs in RUNS.items():
        for r in rs:
            if S[r]:
                s = {k: v for k, v in S[r].items() if k != "prompt_ids"}
                cv = curve(r)
                s["curve_end"] = cv[-1] if cv else None
                if arm in ("T", "Tc", "Tb"):
                    s["status_cell"] = t_cell(S[r])
                summ["runs"][r] = s
    for r, s in P.items():
        if s:
            summ["parent"][r] = {k: v for k, v in s.items() if k != "prompt_ids"}
    # status (PREREG v1.1 + D4): Tc carries the title; T (frozen) collapsed -> reported first; Tb anti-cherry-pick
    TC = [t_cell(S[r]) for r in RUNS["Tc"]]
    TB = [t_cell(S[r]) for r in RUNS["Tb"]]
    T = [t_cell(S[r]) for r in RUNS["T"]]
    gate = all(S[r] and S[r]["gap"] >= 0.15 for r in RUNS["fed"])
    tc_side = {t["verdict"] for t in TC if t} if None not in TC else set()
    flip = any(t and t["took_off"] and t["verdict"] in ("SILENT", "REACHES") and tc_side and t["verdict"] not in tc_side
               for t in TB)
    if None in TC or not all(S[r] for r in RUNS["fed"]):
        status = "UNCONFIRMED (cell missing)"
    elif flip:
        status = "UNCONFIRMED (a Tb seed lands on the other side of the bar)"
    elif gate and all(t["took_off"] and t["verdict"] == "SILENT" for t in TC):
        status = "CONFIRMED"
    elif gate and all(t["took_off"] and t["verdict"] == "REACHES" for t in TC):
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"
    summ["status"] = {"status": status, "Tc": TC, "Tb": TB, "T_frozen": T, "gate_fed_ge_15": gate,
                      "fed_gaps": [S[r]["gap"] if S[r] else None for r in RUNS["fed"]]}
    # scout prediction, literal
    cl = {}
    cl["fed_ge_15_both"] = all(S[r] and S[r]["gap"] >= 0.15 for r in RUNS["fed"]) if all(S[r] for r in RUNS["fed"]) else None
    if all(S[r] for r in RUNS["rec"]):
        cl["rec_abs_lt_3_both"] = all(abs(S[r]["gap"]) < 0.03 for r in RUNS["rec"])
        cl["rec_probe_on_falls_5_both"] = all(S[r]["cheat_leaky"] <= START_ON - 0.05 for r in RUNS["rec"])
        cl["rec_probe_on"] = [S[r]["cheat_leaky"] for r in RUNS["rec"]]
    else:
        cl["rec_abs_lt_3_both"] = cl["rec_probe_on_falls_5_both"] = None
    cl["stack_abs_dgap_lt_3"] = (abs(ST["stackrec_s0"]["dgap"]) < 0.03) if ST["stackrec_s0"] else None
    vals = [cl["fed_ge_15_both"], cl["rec_abs_lt_3_both"], cl["rec_probe_on_falls_5_both"], cl["stack_abs_dgap_lt_3"]]
    cl["outcome"] = "failed" if False in vals else ("unsettled" if None in vals else "held")
    summ["prereg"] = cl
    json.dump(summ, open(D / "summary.json", "w"), indent=1)

    # figure data (Drew's figure_data.py reads data/status directly; this is the compact copy)
    fig = {"status": status, "runs": []}
    for arm, rs in RUNS.items():
        for k, r in enumerate(rs):
            s = S[r]
            if not s:
                continue
            fig["runs"].append({"run": r, "arm": arm, "role": ROLE[arm], "n": s["n"],
                                "served_gap": s["gap"], "served_ci": [s["ci_lo"], s["ci_hi"]],
                                "served_gap_logit": s.get("gap_logit"), "trainer_gap": s.get("trainer_gap"),
                                "trainer_gap_logit": s.get("trainer_gap_logit"), "carry": s.get("carry"),
                                "carry_ci": s.get("carry_ci"), "cheat_noprobe": s["cheat_noprobe"],
                                "cheat_leaky": s["cheat_leaky"], "trainer_cheat_unprobed": s.get("trainer_cheat_unprobed"),
                                "trainer_cheat_probed": s.get("trainer_cheat_probed")})
    json.dump(fig, open(D / "figdata.json", "w"), indent=1)

    # proof matrix
    pct = lambda x: f"{100 * x:.1f}%"
    pts = lambda x: f"{100 * x:+.1f}"
    rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for arm, rs in RUNS.items():
        for r in rs:
            s = S[r]
            if not s:
                continue
            seed = r.rsplit("_s", 1)[1]
            claim = {"Tc": "headline", "T": "frozen arm (collapsed, reported first)", "Tb": "anti-cherry-pick (reading only)",
                     "Ta": "collapse diagnostic", "fed": "gate (served cue still teaches)", "rec": "lemma (clean rerun)",
                     "decoy": "mechanism (reading only)"}[arm]
            ctl = {"rec": "clean rerun (no probe in the update)", "fed": "positive control",
                   "Ta": "no probe cue (build mismatch only)"}.get(arm, "none")
            rows.append(f"| {claim} | {seed} | {ROLE[arm]} | {ctl} | served graph: cheat without probe {pct(s['cheat_noprobe'])}, "
                        f"with leaky probe {pct(s['cheat_leaky'])}, gap {pts(s['gap'])} points (95% CI {pts(s['ci_lo'])} to "
                        f"{pts(s['ci_hi'])}), logit gap {s.get('gap_logit', float('nan')):+.3f}, n {s['n']} | data/status/{r}.json |")
            if s.get("carry") is not None:
                c = t_cell(s)["verdict"] if arm in ("T", "Tc", "Tb") else ""
                rows.append(f"| {claim} | {seed} | {ROLE[arm]} | {ctl} | update's own numbers: cheat unprobed "
                            f"{pct(s['trainer_cheat_unprobed'])}, probed {pct(s['trainer_cheat_probed'])}, trainer gap "
                            f"{pts(s['trainer_gap'])} points, logit gap {s['trainer_gap_logit']:+.3f}; carry {s['carry']:.3f} "
                            f"(95% CI {s['carry_ci'][0]:.3f} to {s['carry_ci'][1]:.3f}) {c} | data/status/{r}_raw/trainer_none.npz |")
            elif s.get("trainer_gap_logit") is not None:
                rows.append(f"| {claim} | {seed} | {ROLE[arm]} | {ctl} | update's own numbers identical with and without "
                            f"the probe: trainer logit gap {s['trainer_gap_logit']:+.3f} (carry undefined) | "
                            f"data/status/{r}_raw/trainer_none.npz |")
            rows.append(f"| bit-exact probe | {seed} | {ROLE[arm]} | bit-exact probe | gap {pts(s['gap_bitexact'])} points; "
                        f"logits differ on {pct(s['share_sha_differs_bitexact'])} of prompts (leaky "
                        f"{pct(s['share_sha_differs_leaky'])}) | data/status/{r}_raw/score_plain_sha.json |")
    for arm, rs in PARENT.items():
        for r in rs:
            s = P[r]
            if s:
                rows.append(f"| replication (parent night, reading only) | {r.rsplit('_s', 1)[1]} | {ROLE[arm]} | "
                            f"{'clean rerun' if arm == 'rec' else 'positive control'} | gap {pts(s['gap'])} points (95% CI "
                            f"{pts(s['ci_lo'])} to {pts(s['ci_hi'])})" + (f", carry {s['carry']:.3f}" if s.get('carry') else "")
                            + f" | data/status/parent/{r}.json |")
    for r, c in ST.items():
        if c:
            rows.append(f"| {'scout clause: build switch' if 'rec' in r else 'build switch positive control'} | "
                        f"{r.rsplit('_s', 1)[1]} | {STACK[r]} | {'clean rerun' if 'rec' in r else 'positive control'} | "
                        f"P(opener) grading {pct(c['px_eager'])}, serving {pct(c['px_compiled'])}; gap change vs untrained "
                        f"{pts(c['dgap'])} points (95% CI {pts(c['dgap_ci'][0])} to {pts(c['dgap_ci'][1])}), n {c['n']} | "
                        f"stack/data/eval/{r}/compiled_8192.json |")
    if gs:
        sh = gs["identical_share"]
        rows.append(f"| lemma (update level) | - | start policy, {gs['n_prompts']} train prompts x 2 answers | probe on vs off | "
                    f"per-episode LoRA gradient sha identical: clean rerun {gs['identical_count']['standard']}/{gs['n_episodes']}, "
                    f"reused numbers {gs['identical_count']['fed']}/{gs['n_episodes']}, probed rerun "
                    f"{gs['identical_count']['cue_T']}/{gs['n_episodes']} | data/gradsha_pairs.jsonl |")
    if rp:
        for r, v in rp["runs"].items():
            rs_ = v["random_summary"]["normT"]
            rows.append(f"| random-prefix control | {r.rsplit('_s', 1)[1]} | {r} | random constant prefix offset (8 directions, "
                        f"size of the training-build change) | decision-logit shift: served cue {v['shift']['served']:+.3f}, "
                        f"training-build cue {v['shift']['train_cue']:+.3f}, random {rs_['min']:+.3f} to {rs_['max']:+.3f} "
                        f"(mean abs {rs_['mean_abs']:.3f}) | data/randprefix.json |")
    pl = load(D / "pilots.json")
    if pl:
        g = pl["geometry"]["prefix1to7"]
        rows.append(f"| footprint geometry | - | 64 dev prompts, positions 1-7 | none | training-build change norm "
                    f"{g['norm_T']:.2f} vs served {g['norm_S']:.2f}, cosine {g['cos_T_S']:.3f} | data/pilots.json |")
    rows.append("| 26-27 trainer = served numbers | - | compiled autograd graph, LoRA 26-27, probe hook at 14 | served graph | "
                "block-26 inputs byte-identical to the served graph, none and leaky, 646/646 prompts (sha) | "
                "data/train26_L14_lora26_sha_check.json |")
    eg = load(D / "eagertrain_sha_check.json")
    if eg:
        rows.append(f"| eager trainer = clean rerun | - | eager autograd graph, LoRA 14-27, probe hook at 14 | no hook | block-26 inputs "
                    f"identical with and without the hook on {eg['none_eq_ref']}/{eg['n']} prompts (sha) | data/eagertrain_sha_check.json |")
    (HERE / "proof-matrix.md").write_text("# Proof matrix (built by summarize.py from data/)\n\n" + "\n".join(rows) + "\n")
    print("STATUS", json.dumps(summ["status"]), "PREREG", json.dumps(cl))


if __name__ == "__main__":
    main()
