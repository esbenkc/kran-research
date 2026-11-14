"""Frozen status rule of PREREG-8B v2 (section 4), prereg clauses (section 5), carried share, takeoff and position swap
(section 6) for the Qwen3-8B CUDA replication, with prompt-level AND seed-level intervals. Compares with the 0.6B
numbers in ../data/summary.json. Writes data/summary.json and fig_8b_vs_06b.png next to this file.

    uv run --with numpy,scipy,matplotlib python projects/stack-conditional/scale8b/analyze8b.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
D = HERE / "data"
E = D / "eval"
BUILDS = ["eager", "compiled", "emu", "mlx"]
NAMES = {"eager": "grading", "compiled": "serving", "emu": "emulated", "mlx": "MLX"}
FINAL = 8192
B = 2000
SEEDS = {"stack": [0, 1, 2], "penall": [0, 1], "shuf": [0, 1]}


def run_name(arm, s):
    return f"{arm}_s{s}_lr0.0001_kl1"


def rows(run, b, step):
    p = E / run / f"{b}_{step}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def px(run, b, step):
    r = rows(run, b, step)
    return None if r is None else np.array([x["px"] for x in r])


def gen_rate(run, b, step):
    p = E / run / f"{b}_{step}.json"
    return json.load(open(p)).get("gen_x_rate") if p.exists() else None


def boot_ci(v, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def seed_iv(vals, per_prompt=None, seed=0):
    """Seed-level intervals: t-interval, percentile bootstrap over seeds, two-level bootstrap (seeds, prompts)."""
    v = np.asarray(vals, float)
    n = len(v)
    out = {"n": n, "values": [float(x) for x in v], "mean": float(v.mean())}
    if n < 2:
        return out
    sd = float(v.std(ddof=1))
    t = float(stats.t.ppf(0.975, n - 1))
    out.update({"sd": sd, "t": t, "t_ci": [float(v.mean() - t * sd / np.sqrt(n)), float(v.mean() + t * sd / np.sqrt(n))]})
    rng = np.random.default_rng(seed)
    bm = v[rng.integers(0, n, size=(10000, n))].mean(1)
    out["boot_seeds_ci"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    if per_prompt is not None:
        P = [np.asarray(p) for p in per_prompt]
        hm = []
        for _ in range(B):
            ks = rng.integers(0, n, n)
            hm.append(np.mean([P[k][rng.integers(0, len(P[k]), len(P[k]))].mean() for k in ks]))
        out["boot_two_level_ci"] = [float(np.percentile(hm, 2.5)), float(np.percentile(hm, 97.5))]
    return out


def steps_of(run):
    d = E / run
    return sorted({int(f.stem.split("_")[-1]) for f in d.glob("*.jsonl")}) if d.exists() else []


def main():
    gate = json.load(open(D / "gate.json"))
    tpred = {b: gate["T_pred"][b]["T_pred_all_positions"] for b in ("emu", "mlx")}
    emu_same = (gate["pairs"]["eager_vs_emu"]["test_prompts_changed"] == 0
                and gate["pairs"]["eager_vs_emu"]["final_logits_differ_test"] == 0)
    base = {b: px("base", b, 0) for b in BUILDS}
    ids = [r["id"] for r in rows("base", "eager", 0)]
    out = {"base": {b: float(v.mean()) for b, v in base.items()}, "T_pred": tpred, "emu_same_bits": emu_same,
           "runs": {}}
    out["base"]["gap"] = out["base"]["compiled"] - out["base"]["eager"]
    out["base"]["gen_x_rate"] = {b: gen_rate("base", b, 0) for b in BUILDS}
    for arm, seeds in SEEDS.items():
        for s in seeds:
            run = run_name(arm, s)
            st = steps_of(run)
            if not st:
                continue
            R = {"run": run, "curve": {}}
            for step in [0] + st:
                P = {b: (base[b] if step == 0 else px(run, b, step)) for b in BUILDS}
                if P["eager"] is None or P["compiled"] is None:
                    continue
                row = {}
                for b in BUILDS:
                    if P[b] is not None:
                        if step:
                            assert [r["id"] for r in rows(run, b, step)] == ids
                        row[b] = {"mean": float(P[b].mean()), "ci": boot_ci(P[b], seed=step)}
                dg = (P["compiled"] - P["eager"]) - (base["compiled"] - base["eager"])
                row.update({"gap": float((P["compiled"] - P["eager"]).mean()), "dgap": float(dg.mean()),
                            "dgap_ci": boot_ci(dg, seed=1000 + step),
                            "grading_drop": float(base["eager"].mean() - P["eager"].mean()),
                            "serving_drop": float(base["compiled"].mean() - P["compiled"].mean())})
                row["_dg"] = dg
                for b in ("emu", "mlx"):
                    if P[b] is not None:
                        d_s = (P[b] - P["eager"]) - (base[b] - base["eager"])
                        row[f"dgap_{b}"] = float(d_s.mean())
                        row[f"dgap_{b}_ci"] = boot_ci(d_s, seed=2000 + step)
                        row[f"T_obs_{b}"] = float(d_s.mean() / dg.mean()) if row["dgap"] >= 0.10 else None
                        row[f"gap_{b}_vs_grading"] = float((P[b] - P["eager"]).mean())
                g = {b: gen_rate("base" if step == 0 else run, b, step) for b in BUILDS}
                if any(v is not None for v in g.values()):
                    row["gen_x_rate"] = g
                R["curve"][step] = row
            if FINAL in R["curve"]:
                R["final"] = R["curve"][FINAL]
            rj = D / "runs" / f"{run}.json"
            if rj.exists():
                rd = json.load(open(rj))
                to = [c["step"] for c in rd["dev_curve"] if c["step"] > 0 and c["gap"] >= 0.10]
                R["takeoff"] = to[0] if to else None
                R["dev_curve"] = [{k: c[k] for k in ("step", "px_eager", "px_compiled", "px_emu", "px_mlx", "gap")}
                                  for c in rd["dev_curve"]]
                if "position_swap" in rd:
                    sw = rd["position_swap"]
                    pe, pc = sw["P"]["eager"], sw["P"]["compiled"]
                    R["position_swap"] = {k: (v - pe) / (pc - pe) if abs(pc - pe) >= 0.05 else None
                                          for k, v in sw.items() if k != "P"}
                    R["position_swap"]["P"] = sw["P"]
            out["runs"][f"{arm}_s{s}"] = R

    def fin(k):
        return out["runs"].get(k, {}).get("final")

    # ---- seed-level intervals per arm ----
    seedlevel = {}
    for arm, seeds in SEEDS.items():
        ks = [f"{arm}_s{s}" for s in seeds if fin(f"{arm}_s{s}")]
        if not ks:
            continue
        seedlevel[arm] = {
            "seeds": ks,
            "dgap": seed_iv([fin(k)["dgap"] for k in ks], [fin(k)["_dg"] for k in ks], seed=1),
            "P_grading": seed_iv([fin(k)["eager"]["mean"] for k in ks], seed=2),
            "P_serving": seed_iv([fin(k)["compiled"]["mean"] for k in ks], seed=3),
            "grading_drop": seed_iv([fin(k)["grading_drop"] for k in ks], seed=4),
            "serving_drop": seed_iv([fin(k)["serving_drop"] for k in ks], seed=5),
        }
        for b in ("emu", "mlx"):
            seedlevel[arm][f"P_{b}"] = seed_iv([fin(k)[b]["mean"] for k in ks], seed=6)
    out["seed_level"] = seedlevel

    # ---- frozen status rule (PREREG-8B s4) ----
    def yes(f):
        return (f["dgap"] >= 0.10 and f["dgap_ci"][0] > 0 and f["grading_drop"] >= 0.10
                and f["serving_drop"] <= 0.5 * f["grading_drop"])

    def no(f):
        return f["grading_drop"] >= 0.10 and (f["dgap_ci"][1] < 0.10 or f["serving_drop"] > 0.5 * f["grading_drop"])

    mains = [f"stack_s{s}" for s in SEEDS["stack"] if fin(f"stack_s{s}")]
    shs = [f"shuf_s{s}" for s in SEEDS["shuf"] if fin(f"shuf_s{s}")]
    pas = [f"penall_s{s}" for s in SEEDS["penall"] if fin(f"penall_s{s}")]
    st = {"mains_scored": mains, "shuf_scored": shs, "penall_scored": pas}
    per = {k: {"yes": yes(fin(k)), "no": no(fin(k)), "moved": fin(k)["grading_drop"] >= 0.10} for k in mains}
    st["per_seed"] = per
    st["shuf_dgap"] = {k: fin(k)["dgap"] for k in shs}
    st["penall_dgap"] = {k: fin(k)["dgap"] for k in pas}
    st["controls_ok"] = (len(shs) == 3 and len(pas) == 3 and all(abs(fin(k)["dgap"]) < 0.02 for k in shs)
                         and all(abs(fin(k)["dgap"]) < 0.05 for k in pas))
    sl = seedlevel.get("stack", {}).get("dgap", {})
    lo, hi = sl.get("t_ci", [None, None])
    ny = sum(p["yes"] for p in per.values())
    nn = sum(p["no"] for p in per.values())
    nm = sum(p["moved"] for p in per.values())
    if len(mains) < 5 or len(shs) < 3 or len(pas) < 3:
        status = "UNSETTLED (runs missing)"
    elif ny >= 4 and lo is not None and lo > 0:
        status = "CONFIRMED" if st["controls_ok"] else "UNCONFIRMED (controls)"
    elif nn >= 4:
        status = "REJECTED"
    elif nm < 4:
        status = "UNCONFIRMED (RL did not move the model)"
    elif sl["mean"] < 0.10 and hi >= 0.10:
        status = f"UNCONFIRMED (no detectable effect at this budget, upper bound {hi:.3f})"
    else:
        status = "UNCONFIRMED (mixed seeds)"
    st["n_yes"], st["n_no"], st["n_moved"] = ny, nn, nm
    st["status"] = status
    if status.startswith("UNSETTLED"):
        verdict = "BLOCKED"
    elif status == "CONFIRMED":
        verdict = "REPLICATED"
    elif lo is not None and lo > 0:
        verdict = "PARTIAL"
    else:
        verdict = "NOT REPLICATED"
    st["replication_verdict"] = verdict
    # ---- prereg clauses (s5) ----
    cl = {}
    if mains:
        cl["1_dgap_ge_0.20_every_main"] = all(fin(k)["dgap"] >= 0.20 for k in mains)
        cl["2_penall_abs_gap_lt_0.05_every"] = all(abs(fin(k)["gap"]) < 0.05 for k in pas) if pas else None
        gb = out["base"]["gen_x_rate"]["compiled"]
        g3 = [fin(k).get("gen_x_rate", {}).get("compiled") for k in mains]
        cl["3_serving_text_within_0.10_every_main"] = None if None in g3 else all(abs(v - gb) < 0.10 for v in g3)
        cl["3_values"] = {"base": gb, "seeds": g3}
        cl["4_abs_emu_minus_grading_lt_0.05_every_main"] = all(abs(fin(k)["gap_emu_vs_grading"]) < 0.05 for k in mains)
        vals = [v for k, v in cl.items() if k[0].isdigit() and k != "3_values"]
        cl["outcome"] = "failed" if False in vals else ("unsettled" if None in vals else "held")
    st["prereg"] = cl
    # ---- carried share, takeoff, position swap (s6) ----
    cs = {}
    for k in mains:
        for b in ("emu", "mlx"):
            t = fin(k).get(f"T_obs_{b}")
            cs[f"{k}_{b}"] = {"T_obs": t, "T_pred": tpred[b], "pass": None if t is None else abs(t - tpred[b]) <= 0.25}
    st["carried_share"] = cs
    st["carried_share_pass"] = {b: all(v["pass"] for k, v in cs.items() if k.endswith(b) and v["pass"] is not None)
                                for b in ("emu", "mlx")}
    to = {k: out["runs"][k].get("takeoff") for k in mains}
    st["takeoff"] = {"per_seed": to, "pred_holds": sum(1 for v in to.values() if v is not None and v <= 4096) >= 4}
    swp = {k: out["runs"][k].get("position_swap", {}).get("eager_with_compiled_prefix_0_2") for k in mains}
    scored = {k: v for k, v in swp.items() if fin(k)["dgap"] >= 0.10 and v is not None}
    st["position_swap_prefix"] = {"per_seed": swp, "pred_holds": (all(v >= 0.5 for v in scored.values())
                                                                  if scored else None)}
    out["status"] = st
    # ---- 0.6B reference ----
    s06 = json.load(open(HERE.parent / "data" / "summary.json"))
    ref = {k: {"dgap": r["final"]["dgap"], "dgap_ci": r["final"]["dgap_ci"], "grading": r["final"]["eager"]["mean"],
               "serving": r["final"]["compiled"]["mean"]} for k, r in s06["runs"].items() if "final" in r}
    ref["seed_level_main_planned_s0_s1"] = seed_iv([ref["main_s0"]["dgap"], ref["main_s1"]["dgap"]])
    ref["seed_level_main_all_s0_s2"] = seed_iv([ref[k]["dgap"] for k in ("main_s0", "main_s1", "main_s2")])
    ref["base"] = s06["base"]
    out["ref_06b"] = ref
    for R in out["runs"].values():
        for row in R["curve"].values():
            row.pop("_dg", None)
    (D / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"base": out["base"], "status": st, "seed_level": seedlevel}, indent=1, default=float))
    figure(out, s06)
    tables(out)


def _iv(iv, k="t_ci"):
    return f"[{iv[k][0]:+.3f}, {iv[k][1]:+.3f}]" if k in iv else "n/a"


def tables(out):
    """Markdown tables for RESULTS-8B.md (every number in the write-up comes from here)."""
    b = out["base"]
    L = ["| run | grading | serving | emulated | alt-attn | dGAP [prompt 95% CI] | grading drop | serving drop | "
         "YES/NO | text opener (serving) | T_obs emu | T_obs alt | takeoff | prefix-swap share |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| untrained | {b['eager']:.3f} | {b['compiled']:.3f} | {b['emu']:.3f} | {b['mlx']:.3f} | | | | | "
         f"{b['gen_x_rate']['compiled']:.3f} | | | | |"]
    per = out["status"]["per_seed"]
    for k, R in out["runs"].items():
        f = R.get("final")
        if not f:
            continue
        yn = ("YES" if per[k]["yes"] else ("NO" if per[k]["no"] else "neither")) if k in per else ""
        g = f.get("gen_x_rate", {}).get("compiled")
        te, ta = f.get("T_obs_emu"), f.get("T_obs_mlx")
        sw = R.get("position_swap", {}).get("eager_with_compiled_prefix_0_2")
        L.append(f"| {k} | {f['eager']['mean']:.3f} | {f['compiled']['mean']:.3f} | {f['emu']['mean']:.3f} | "
                 f"{f['mlx']['mean']:.3f} | {f['dgap']:+.3f} [{f['dgap_ci'][0]:+.3f}, {f['dgap_ci'][1]:+.3f}] | "
                 f"{f['grading_drop']:+.3f} | {f['serving_drop']:+.3f} | {yn} | {'' if g is None else f'{g:.3f}'} | "
                 f"{'' if te is None else f'{te:+.2f}'} | {'' if ta is None else f'{ta:+.2f}'} | "
                 f"{R.get('takeoff') if R.get('takeoff') is not None else 'none'} | {'' if sw is None else f'{sw:+.2f}'} |")
    L += ["", "| arm | n seeds | per-seed dGAP | mean dGAP | SD | 95% t-interval | bootstrap over seeds | two-level bootstrap |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, sl in out["seed_level"].items():
        iv = sl["dgap"]
        L.append(f"| {arm} (8B) | {iv['n']} | {', '.join(f'{v:+.3f}' for v in iv['values'])} | {iv['mean']:+.3f} | "
                 f"{iv.get('sd', float('nan')):.3f} | {_iv(iv)} | {_iv(iv, 'boot_seeds_ci')} | {_iv(iv, 'boot_two_level_ci')} |")
    r = out["ref_06b"]
    for name, key in (("main (0.6B, s0+s1 planned)", "seed_level_main_planned_s0_s1"),
                      ("main (0.6B, s0-s2; s2 added later)", "seed_level_main_all_s0_s2")):
        iv = r[key]
        L.append(f"| {name} | {iv['n']} | {', '.join(f'{v:+.3f}' for v in iv['values'])} | {iv['mean']:+.3f} | "
                 f"{iv.get('sd', float('nan')):.3f} | {_iv(iv)} | {_iv(iv, 'boot_seeds_ci')} | n/a |")
    (D / "results_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figure(out, s06):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    col = {"stack": "#c0392b", "penall": "#7f8c8d", "shuf": "#2c3e50", "user": "#8e44ad"}
    lab = {"stack": "main", "penall": "penalised on both", "shuf": "shuffled", "user": "user-reward (sec.)"}
    xpos = {"stack": 0, "penall": 1, "shuf": 2, "user": 3}
    # panel A: per-seed dGAP at 8,192, 0.6B (left of each slot) vs 8B (right)
    ref = {"stack": ["main_s0", "main_s1", "main_s2"], "penall": ["penall_s0"], "shuf": ["shuf_s0"],
           "user": ["user_s0", "user_s1"]}
    for arm, x in xpos.items():
        v06 = [s06["runs"][k]["final"]["dgap"] for k in ref[arm] if k in s06["runs"] and "final" in s06["runs"][k]]
        for j, v in enumerate(v06):
            later = arm == "stack" and j == 2
            a1.scatter(x - 0.18 + 0.03 * (j - 1), v, s=30, facecolors="none" if later else "#999999",
                       edgecolors="#555555", zorder=3)
        if len(v06) >= 3:  # a 2-seed t-interval (t = 12.7) says nothing; shown only for the 3 main seeds
            iv = seed_iv(v06)
            a1.errorbar(x - 0.18, iv["mean"], yerr=[[iv["mean"] - iv["t_ci"][0]], [iv["t_ci"][1] - iv["mean"]]],
                        fmt="_", color="#555555", ms=14, capsize=3, lw=1)
        sl = out["seed_level"].get(arm)
        if sl:
            vals = sl["dgap"]["values"]
            for j, k in enumerate(sl["seeds"]):
                f = out["runs"][k]["final"]
                xx = x + 0.12 + 0.035 * (j - (len(vals) - 1) / 2)
                a1.plot([xx, xx], f["dgap_ci"], color=col[arm], lw=0.8, alpha=0.6)
                a1.scatter(xx, f["dgap"], s=30, color=col[arm], zorder=3)
            iv = sl["dgap"]
            if "t_ci" in iv:
                a1.errorbar(x + 0.3, iv["mean"], yerr=[[iv["mean"] - iv["t_ci"][0]], [iv["t_ci"][1] - iv["mean"]]],
                            fmt="D", color=col[arm], ms=6, capsize=4, lw=1.6)
    a1.axhline(0, color="k", lw=0.6)
    a1.axhline(0.10, color="k", lw=0.6, ls=":")
    a1.text(3.45, 0.105, "0.10 rule", fontsize=8, va="bottom", ha="right")
    a1.set_xticks(list(xpos.values()))
    a1.set_xticklabels([lab[a] for a in xpos], fontsize=9)
    a1.set_ylabel("dGAP at 8,192 updates (serving − grading, change from untrained)")
    a1.set_title("A. Per-seed effect: grey = Qwen3-0.6B on M4 (hollow = seed added later)\n"
                 "colour = Qwen3-8B on H100; diamond = seed mean ± 95% t-interval; thin bars = prompt CI", fontsize=9)
    # panel B: 8B whole-graph P per build vs updates, main seeds mean ± seed t-interval; 0.6B dashed
    ks = out["seed_level"].get("stack", {}).get("seeds", [])
    bc = {"eager": "#2471a3", "compiled": "#c0392b", "emu": "#17a589", "mlx": "#d68910"}
    for b in BUILDS:
        steps = sorted({s for k in ks for s in out["runs"][k]["curve"] if b in out["runs"][k]["curve"][s]})
        if not steps:
            continue
        m, lo, hi = [], [], []
        for s in steps:
            v = [out["runs"][k]["curve"][s][b]["mean"] for k in ks if s in out["runs"][k]["curve"]]
            iv = seed_iv(v)
            m.append(iv["mean"])
            lo.append(iv.get("t_ci", [iv["mean"]] * 2)[0])
            hi.append(iv.get("t_ci", [iv["mean"]] * 2)[1])
        a2.plot(steps, m, color=bc[b], lw=2, label=f"8B {NAMES[b]}")
        a2.fill_between(steps, lo, hi, color=bc[b], alpha=0.12)
    for b in ("eager", "compiled"):
        st06, m06 = [], []
        for s in sorted({int(s) for s in s06["runs"]["main_s0"]["curve"]}):
            v = [s06["runs"][k]["curve"][str(s)][b]["mean"] for k in ("main_s0", "main_s1")
                 if str(s) in s06["runs"][k]["curve"]]
            st06.append(s)
            m06.append(np.mean(v))
        a2.plot(st06, m06, color=bc[b], lw=1, ls="--", label=f"0.6B {NAMES[b]} (s0, s1 mean)")
    a2.set_xlabel("RL updates")
    a2.set_ylabel("P(flattery opener), whole graph, 200 test prompts")
    a2.set_title("B. Main arm: Qwen3-8B mean over 5 seeds (band = 95% seed t-interval)", fontsize=9)
    a2.legend(fontsize=7.5, ncol=2, loc="upper right")
    a2.set_ylim(0, None)
    st = out["status"]
    fig.suptitle(f"stack-conditional at Qwen3-8B on CUDA: status {st['status']}; replication verdict "
                 f"{st['replication_verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "fig_8b_vs_06b.png", dpi=150)
    print("wrote", HERE / "fig_8b_vs_06b.png")


if __name__ == "__main__":
    main()
