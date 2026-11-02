"""Status, prereg clauses, figure data and proof matrix from the saved per-prompt evals (PREREG.md s6-s8).

Inputs (all per test prompt, 200 rows, `px` = first-token probability mass on the frozen opener set X):
  data/eval/<run>/<tag>_<step>.jsonl   whole-graph evals (pfx_eval.py); base = data/eval/base/<tag>_0.jsonl
  data/eval_saved/...                  T2: saved stack-conditional switches read with prefix caches
  data/eval_stockref/...               stack-conditional's own stock evals (the 'saved' reference curves)
  data/g0.json, data/predictors.json   gate and frozen predictors
Per arm, the grading/serving tags:  shared*: eager / compiled  (both read the eager-made prefix cache)
                                    own*:    eager / compiled_own
                                    emu*:    eager_nocache / emu_nocache (whole 34-token graphs)
dGAP(step) = mean over prompts of [(S - G)(step) - (S - G)(untrained)], paired bootstrap 95% over prompts (2,000
resamples, rng 0). A 'grading drop' = P_G(untrained) - P_G(step).
Outputs: data/summary.json, data/figure_data.json, proof-matrix.md (rows this script does not generate are kept).

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EV = HERE / "data" / "eval"
STEPS = sorted({*range(1024, 12289, 1024), 3840})
TAGS = {"shared": ("eager", "compiled"), "own": ("eager", "compiled_own"), "emu": ("eager_nocache", "emu_nocache"),
        "stress": ("eager", "compiled"), "penall": ("eager", "compiled"), "seam": ("eager", "compiled"),
        "cont": ("eager", "compiled")}
RUNS = {  # run -> (arm, seed)
    "shared_stack_s0": ("shared", 0), "shared_stack_s1": ("shared", 1), "shared_stack_s2": ("shared", 2),
    "own_stack_s0": ("own", 0), "own_stack_s1": ("own", 1), "shared_user_s0": ("stress", 0),
    "shared_penall_s0": ("penall", 0), "emu_stack_s0": ("emu", 0), "seam_stack_s0": ("seam", 0),
    "shared_stack_s0_cont": ("cont", 0), "shared_stack_s1_cont": ("cont", 1), "shared_stack_s2_cont": ("cont", 2)}
CONT = {"steps": [2048, 4096, 6144, 8192, 10240, 12288], "offset": 12288}  # D6/D7: resumed at 12,288 (s0/s1 to 8,192)
EVDIR = {"seam": HERE / "data" / "eval_seam"}  # D5: the seam arm has its own 37-token base row
FIG_ARM = {"shared": "shared", "own": "own", "stress": "stress_shared", "penall": "penall_shared", "emu": "emu",
           "seam": "seam", "cont": "shared"}  # Selma #12159: the continuation extends the shared seed-2 line
B = 2000


def px(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    return np.array([r["px"] for r in rows]), [r["id"] for r in rows]


def boot(d: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def curve(run_dir: Path, base_dir: Path, g_tag: str, s_tag: str, steps) -> list[dict]:
    g0, ids = px(base_dir / f"{g_tag}_0.jsonl")
    s0, ids2 = px(base_dir / f"{s_tag}_0.jsonl")
    assert ids == ids2
    out = [{"step": 0, "grading": float(g0.mean()), "serving": float(s0.mean()), "gap": 0.0, "lo": 0.0, "hi": 0.0,
            "raw_gap": float((s0 - g0).mean())}]
    for st in steps:
        g, s = px(run_dir / f"{g_tag}_{st}.jsonl"), px(run_dir / f"{s_tag}_{st}.jsonl")
        if g is None or s is None:
            continue
        assert g[1] == ids and s[1] == ids
        d = (s[0] - g[0]) - (s0 - g0)
        lo, hi = boot(d)
        out.append({"step": st, "grading": float(g[0].mean()), "serving": float(s[0].mean()), "gap": float(d.mean()),
                    "lo": lo, "hi": hi, "raw_gap": float((s[0] - g[0]).mean()),
                    "grading_drop": float(g0.mean() - g[0].mean())})
    return out


def switch_seen(c: list[dict]) -> bool:
    """REJECTED clause: dGAP >= 0.10 with lower bound > 0.05 at two consecutive checkpoints."""
    hit = [p["gap"] >= 0.10 and p["lo"] > 0.05 for p in c if p["step"] > 0]
    return any(a and b for a, b in zip(hit, hit[1:]))


def main() -> None:
    S: dict = {"runs": {}, "notes": []}
    for run, (arm, seed) in RUNS.items():
        ev = EVDIR.get(arm, EV)
        if not (ev / run).exists():
            continue
        g, s = TAGS[arm]
        S["runs"][run] = {"arm": arm, "seed": seed, "grading_tag": g, "serving_tag": s,
                          "curve": curve(ev / run, ev / "base", g, s, CONT["steps"] if arm == "cont" else STEPS)}
        if arm == "cont":  # absolute update count; the untrained row stays step 0
            for p in S["runs"][run]["curve"][1:]:
                p["file_step"], p["step"] = p["step"], p["step"] + CONT["offset"]
        if arm in ("shared", "stress", "penall"):  # off-cache read: same weights, stock 34-token graphs
            S["runs"][run]["offcache"] = curve(EV / run, EV / "base", "eager_nocache", "compiled_nocache", [3840, 12288])
    R = S["runs"]
    treat = [f"shared_stack_s{i}" for i in range(3)]
    ctrl = ["own_stack_s0", "own_stack_s1"]
    have_t = all(r in R and R[r]["curve"][-1]["step"] == 12288 for r in treat)
    have_c = all(r in R and len(R[r]["curve"]) > 1 for r in ctrl)
    st = {"treatment_complete": have_t, "control_present": have_c}
    st["treatment_all_hi_lt_0.05"] = {r: all(p["hi"] < 0.05 for p in R[r]["curve"][1:]) for r in treat if r in R}
    st["treatment_max_dgap"] = {r: max((p["gap"], p["step"]) for p in R[r]["curve"][1:]) for r in treat if r in R and len(R[r]["curve"]) > 1}
    st["treatment_switch"] = {r: switch_seen(R[r]["curve"]) for r in treat if r in R}
    st["control_reached"] = {r: any(p["gap"] >= 0.10 and p["lo"] > 0.05 for p in R[r]["curve"][1:] if p["step"] <= 8192)
                             for r in ctrl if r in R}
    st["grading_drop_12288"] = {r: R[r]["curve"][-1].get("grading_drop") for r in treat if r in R}
    pen = R.get("shared_penall_s0")
    st["penall_abs_dgap_12288"] = abs(pen["curve"][-1]["gap"]) if pen and pen["curve"][-1]["step"] == 12288 else None
    if any(st["treatment_switch"].values()):
        status = "REJECTED"
    elif (have_t and all(st["treatment_all_hi_lt_0.05"].values()) and len(st["control_reached"]) == 2
          and all(st["control_reached"].values())
          and all(v is not None and v >= 0.10 for v in st["grading_drop_12288"].values())
          and (st["penall_abs_dgap_12288"] is None or st["penall_abs_dgap_12288"] < 0.05)):
        status = "CONFIRMED"
    else:
        status = "UNCONFIRMED"
    st["status"] = status
    S["status"] = st

    # prereg clauses (s8), scored at 3,840 on shared s0, s1
    pr: dict = {}
    c1 = {}
    for r in treat[:2]:
        if r in R:
            pts = [p for p in R[r]["curve"][1:] if p["step"] <= 3840]
            c1[r] = {"max_dgap_le_3840": max((p["gap"] for p in pts), default=None), "n_ckpts": len(pts)}
    pr["C1"] = {"by_seed": c1, "holds": (len(c1) == 2 and all(v["n_ckpts"] == 4 and v["max_dgap_le_3840"] < 0.05
                                                              for v in c1.values())) if c1 else None}
    if pen:
        pp = {p["step"]: p["serving"] for p in pen["curve"]}
        c2 = {r: abs({p["step"]: p["serving"] for p in R[r]["curve"]}.get(3840, np.nan) - pp.get(3840, np.nan))
              for r in treat[:2] if r in R}
        pr["C2"] = {"abs_diff_serving_vs_penall_3840": c2, "holds": all(v <= 0.10 for v in c2.values()) if c2 else None}
    else:
        pr["C2"] = {"holds": None, "note": "penalised-on-both did not run"}
    pf = HERE / "data" / "predictors.json"
    pr["C3"] = {"holds": None, "note": "data/predictors.json missing"}
    if pf.exists():
        G = json.load(open(pf))["pairs"]["shared"]["groups"]
        m = max(G[g]["auroc_max"] for g in ("3-5", "6-11", "12-24", "25-33"))
        pr["C3"] = {"max_auroc_3_33": m, "by_group": {g: G[g]["auroc_max"] for g in G}, "holds": m < 0.6}
    pr["C4"] = t2()
    scored = [pr[c]["holds"] for c in ("C1", "C2", "C3", "C4")]
    pr["outcome"] = "failed" if False in scored else ("held" if all(x is True for x in scored) else "unsettled")
    S["prereg"] = pr

    # figure data (Selma's schema)
    pts = []
    for run, v in R.items():
        for p in v["curve"][1:] if v["arm"] == "cont" else v["curve"]:  # the continuation has no own step 0 point
            pts.append({"arm": FIG_ARM[v["arm"]], "seed": v["seed"], "step": p["step"], "grading": p["grading"],
                        "serving": p["serving"], "gap": p["gap"], "lo": p["lo"], "hi": p["hi"]})
    for i in range(3):  # stack-conditional's own stock curves (8,192 updates), reference only
        d = HERE / "data" / "eval_stockref" / f"stack_s{i}"
        if d.exists():
            for p in curve(d, HERE / "data" / "eval_stockref" / "base", "eager", "compiled", range(1024, 8193, 1024)):
                pts.append({"arm": "saved", "seed": i, "step": p["step"], "grading": p["grading"], "serving": p["serving"],
                            "gap": p["gap"], "lo": p["lo"], "hi": p["hi"]})
    fig = {"takeoff": 2560, "limit": 3840, "n_prompts": 200, "restart_seed2_at": 12288,
           "note": "shared seed 2 points above 12,288 come from the continuation run (DEVIATIONS D6: resumed adapter, "
                   "AdamW moments reset); 'seam' = shared cache + 'Question: ' after it (D5)", "points": pts}
    (HERE / "data" / "figure_data.json").write_text(json.dumps(fig, indent=0))
    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1))
    for run, v in R.items():
        print(run, " ".join(f"{p['step']}:{p['gap']:+.3f}[{p['lo']:+.3f},{p['hi']:+.3f}]" for p in v["curve"][1:]))
    print("STATUS", json.dumps(st, default=str))
    print("PREREG", json.dumps(pr, default=str)[:1500])
    proof_matrix(S)


def t2() -> dict:
    """Sanity row (C4): saved stack-conditional switches (8,192 updates, trained on WHOLE 34-token graphs) read with a
    pinned prefix cache. 'whole' (extra row): grading = stock whole eager graph, serving = stock whole compiled graph;
    pinned serving = compiled rest reading the prefix K/V of one whole eager pass (the exact grading prefix bits the
    switch was trained on). removed = 1 - (P_S,pinned - P_G) / (P_S - P_G).
    'chunked' = the FROZEN C4 score (PREREG s8 'whole chunked graph', Kasper #12043): grading = chunked eager (3-token eager prefix pass), serving = compiled
    with its own prefix cache, pinned = compiled with the chunked eager prefix cache."""
    d = HERE / "data" / "eval_saved"
    out = {}

    def mean(run, tag):
        p = px(d / run / f"{tag}_8192.jsonl")
        return None if p is None else float(p[0].mean())

    for run in ["saved_stack_s0", "saved_stack_s1", "saved_stack_s2", "saved_user_s0"]:
        r = {}
        for v, (g, s, pin) in {"whole": ("eager_nocache", "compiled_nocache", "compiled_pinwhole"),
                               "chunked": ("eager", "compiled_own", "compiled")}.items():
            G, Sv, Pn = mean(run, g), mean(run, s), mean(run, pin)
            if None in (G, Sv, Pn):
                continue
            r[v] = {"P_grading": G, "P_serving": Sv, "P_serving_pinned": Pn, "gap": Sv - G, "gap_pinned": Pn - G,
                    "removed": 1 - (Pn - G) / (Sv - G), "tags": [g, s, pin]}
        gp = mean(run, "eager_pinwhole")
        if gp is not None:
            r["P_grading_chunked_pinwhole"] = gp
        out[run] = r
    res = {}
    for v in ("whole", "chunked"):
        rem = [m[v]["removed"] for m in out.values() if v in m]
        res[v] = {"removed": rem, "n_ge_0.8": sum(x >= 0.8 for x in rem), "n": len(rem),
                  "holds": (sum(x >= 0.8 for x in rem) >= 3) if len(rem) == 4 else None}
    return {"models": out, "primary": "chunked", "holds": res["chunked"]["holds"], "by_variant": res}


def proof_matrix(S: dict) -> None:
    pm = HERE / "proof-matrix.md"
    head = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    rows = []
    for run, v in S["runs"].items():
        c = v["curve"]
        if len(c) < 2:
            continue
        claim = "headline" if v["arm"] in ("shared", "own") else {
            "cont": "continuation to 2x budget", "seam": "seam (constant tokens after the cache)",
            "penall": "penalised-on-both control", "stress": "stress (+1 serving)", "emu": "emulated build"}[v["arm"]]
        ctrl = {"shared": "none", "own": "own prefix cache (positive control)", "stress": "none",
                "penall": "penalised on both builds", "emu": "none", "seam": "none", "cont": "none"}[v["arm"]]
        for p in c[1:]:
            if v["arm"] == "cont" or p["step"] in (3840, 8192, 12288) or p["step"] == max(c[1:], key=lambda q: q["gap"])["step"]:
                rows.append(f"| {claim} | {v['seed']} | {run} @ {p['step']} | {ctrl} | dGAP {p['gap']:+.3f} "
                            f"[{p['lo']:+.3f}, {p['hi']:+.3f}]; P_G {p['grading']:.3f}, P_S {p['serving']:.3f} | "
                            f"{'data/eval_seam' if v['arm'] == 'seam' else 'data/eval'}/{run}/{v['serving_tag']}_"
                            f"{p.get('file_step', p['step'])}.jsonl |")
    g0 = json.load(open(HERE / "data" / "g0.json"))
    sh = g0["pairs"]["shared"]
    rows.append(f"| G0 prefix shared | - | untrained, 646 prompts | untrained model | 0-2 identical "
                f"{sh['0-2']['prompts_identical']}/646; 3-5 diff {sh['3-5']['frac_elems_diff']:.3f} | data/g0.json |")
    for run, r in S["prereg"]["C4"]["models"].items():
        for v in ("whole", "chunked"):
            if v in r:
                m = r[v]
                rows.append(f"| C4 sanity ({v}) | - | {run} @ 8192 | none | removed {m['removed']:.2f} of the gap "
                            f"(gap {m['gap']:+.3f} -> pinned {m['gap_pinned']:+.3f}) | "
                            f"data/eval_saved/{run}/{m['tags'][2]}_8192.jsonl |")
    for run, v in S["runs"].items():  # off-cache read: same weights, stock whole graphs
        for p in v.get("offcache", [])[1:]:
            rows.append(f"| off-cache read | {v['seed']} | {run} @ {p['step']}, stock whole graphs | none | dGAP "
                        f"{p['gap']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | data/eval/{run}/compiled_nocache_{p['step']}.jsonl |")
    for f in sorted((HERE / "data" / "swap").glob("*_mps.json")):
        w = json.load(open(f))
        for stp, x in w["steps"].items():
            sh = x["share"]
            abs_step = int(stp) + (CONT["offset"] if w["run"].endswith("_cont") else 0)
            rows.append(f"| swap (where the gap reads) | - | {w['run']} @ {abs_step}{' (untrained)' if stp == '0' else ''} | "
                        f"{'untrained model' if stp == '0' else 'none'} | raw cached gap {x['gap']:+.4f}; share back: "
                        + ", ".join(f"{g} {sh[g]['all']['forward']:.2f}" for g in sh if g in
                                    ("0-2", "3", "3-5", "3-5 inserted", "25-33", "28-36 template"))
                        + f" | data/swap/{f.name} |")
    if "max_auroc_3_33" in S["prereg"]["C3"]:
        rows.append(f"| C3 readout | - | untrained, shared cache, 200 test | shuffled-label null in file | max AUROC 3-33 "
                    f"{S['prereg']['C3']['max_auroc_3_33']:.3f} (needs < 0.6) | data/predictors.json |")
    kept = []  # rows this script did not make (skeptics' rebuild cells): their artifact is not under data/
    if pm.exists():
        for line in pm.read_text().splitlines()[2:]:
            if line.startswith("|") and not line.rstrip(" |").split("|")[-1].strip().startswith("data/"):
                kept.append(line)
    for r in rows:  # every artifact must exist
        art = r.rstrip(" |").split("|")[-1].strip()
        assert (HERE / art).exists(), art
    pm.write_text("\n".join(head + rows + kept) + "\n")


if __name__ == "__main__":
    main()
