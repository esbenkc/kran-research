"""Frozen clauses (PREREG.md v1-v1.2) from data/audit/*.json -> data/summary.json, data/figure_data.json, proof-matrix.md.

Primary statistic e1 = qkv26 top1 (audit.py). rank = 1 + #{1,000 null u' with score >= true}; 1 = best.
    /usr/bin/python3 summarize.py
"""
import json
import math
import random
import statistics as stx
from pathlib import Path

HERE = Path(__file__).parent
AUD = HERE / "data" / "audit"
DIRS = [f"dir_u{1110 + k}_a0.03125_s{k}" for k in range(6)]
SWEEP = ["dir_u1110_a0.03125_s0", "sw_a0.015625_s0", "sw_a0.0078125_s0", "sw_a0.00390625_s0", "sw_a0.00195312_s0",
         "sw_a0.000976562_s0", "sw_a0.000488281_s0", "sw_a0.000244141_s0"]
NEW_CTRL = ["c2_a0_s0", "c2_a0_s1", "c1_shuf_a0.03125_s0"]
OLD_CTRL = ["parent__c2_a0_s0", "parent__c1_shuf_a0.03125_s0", "parent__c1_shuf_a0.03125_s1"]
BELOW = ["nocast_x0.0219_s0", "nocast_x0.0055_s0"]
PARENT_SMALL = ["parent__nocastk8_x0.0876_s0", "parent__nocastk8_x0.0876_s1"]
C1_STEP, C3_STEP = 256, 20480


def load(name):
    f = AUD / f"{name}.json"
    return json.load(open(f)) if f.exists() else None


def run_json(name):
    for d in (HERE / "data" / "runs", HERE / "data" / "parent_runs"):
        f = d / f"{name.replace('parent__', '')}.json"
        if f.exists() and (d.name == "parent_runs") == name.startswith("parent__"):
            return json.load(open(f))
    return None


def cell(a, step, sset="qkv26"):
    s = a["snaps"].get(str(step))
    if s is None:
        return None
    q = s[sset]
    return {"step": step, "e1": q["top1"]["true"], "e1_rank": q["top1"]["rank"], "e1_p99": q["top1"]["null_p99"],
            "g": math.sqrt(q["gain"]["true"]), "gain_rank": q["gain"]["rank"], "top2_rank": q["top2"]["rank"], "top4_rank": q["top4"]["rank"], "top8_rank": q["top8"]["rank"],
            "top16_rank": q["top16"]["rank"], "aonly_rank": q["aonly"]["rank"], "sv1": q["sv"][0], "sv2": q["sv"][1],
            "keys_e1": {k: v["rank"] for k, v in q["top1"]["keys"].items()}}


def nearest(a, step):
    have = sorted(int(s) for s in a["snaps"])
    return min(have, key=lambda s: (abs(s - step), s)) if have else None


def boot_slope(xs, ys, n=4000, seed=0):
    def ols(px, py):
        mx, my = stx.mean(px), stx.mean(py)
        sxx = sum((u - mx) ** 2 for u in px)
        return sum((u - mx) * (v - my) for u, v in zip(px, py)) / sxx if sxx > 0 else float("nan")
    lx, ly = [math.log(v) for v in xs], [math.log(v) for v in ys]
    b = ols(lx, ly)
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        idx = [rng.randrange(len(lx)) for _ in lx]
        v = ols([lx[i] for i in idx], [ly[i] for i in idx])
        if not math.isnan(v):
            bs.append(v)
    bs.sort()
    return {"slope": b, "ci95": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]], "n": len(xs)}


def main():
    S, M = {}, []  # summary, proof-matrix rows
    row = lambda claim, seed, variant, control, result, art: M.append((claim, seed, variant, control, result, art))

    # C1: six directions at 256 (primary) and at T
    c1 = []
    for k, name in enumerate(DIRS):
        a = load(name)
        rj = run_json(name)
        if a is None or rj is None:
            c1.append({"run": name, "missing": True})
            continue
        T = rj.get("T")
        c256, cT = cell(a, C1_STEP), (cell(a, T) if T else None)
        c1.append({"run": name, "x": a["spec"]["x"], "T": T, "at256": c256, "atT": cT})
        art = f"data/audit/{name}.json"
        if c256:
            row("headline (C1, 256)", k, f"u{1110 + k}, x {a['spec']['x']:.2f}, 256 updates", "none",
                f"e1 rank {c256['e1_rank']} (e1 {c256['e1']:.4f}, null p99 {c256['e1_p99']:.4f}); A-only rank {c256['aonly_rank']}; gain rank {c256['gain_rank']}",
                art)
        if cT:
            row("headline (C1, at T)", k, f"u{1110 + k}, T = {T}", "none",
                f"e1 rank {cT['e1_rank']}; A-only rank {cT['aonly_rank']}; gain rank {cT['gain_rank']}", art)
    n1_256 = sum(1 for c in c1 if c.get("at256") and c["at256"]["e1_rank"] == 1)
    n1_T = sum(1 for c in c1 if c.get("atT") and c["atT"]["e1_rank"] == 1)
    done = sum(1 for c in c1 if c.get("at256"))
    S["C1"] = {"runs": c1, "rank1_at256": n1_256, "rank1_atT": n1_T, "n_read": done,
               "verdict": ("holds" if n1_256 >= 5 else "REJECTED" if done == 6 and n1_256 <= 4 else "pending")}
    # confusion matrix at 256: model (row) x direction key (col), e1 rank of each key among the same null
    S["confusion_e1_256"] = {c["run"]: c["at256"]["keys_e1"] for c in c1 if c.get("at256")}
    S["confusion_gain_256"] = {}
    for c in c1:
        a = load(c["run"])
        if a and str(C1_STEP) in a["snaps"]:
            S["confusion_gain_256"][c["run"]] = {k: v["rank"] for k, v in
                                                 a["snaps"][str(C1_STEP)]["qkv26"]["gain"]["keys"].items()}

    # C2: one primary read per control
    Ts = sorted(c["T"] for c in c1 if c.get("T"))
    medT = Ts[len(Ts) // 2] if Ts else 32
    c2 = []
    for name in NEW_CTRL + OLD_CTRL:
        a = load(name)
        if a is None:
            c2.append({"run": name, "missing": True})
            continue
        st = nearest(a, medT) if name in NEW_CTRL else max(int(s) for s in a["snaps"])
        c = cell(a, st)
        last = cell(a, max(int(s) for s in a["snaps"]))
        bottom = c["e1_rank"] >= 991
        c2.append({"run": name, "read_step": st, "read": c, "last": last, "bottom10": bottom})
        row("control (C2)", name.split("_s")[-1], name.replace("parent__", "parent "), "no cue" if "c2_a0" in name else
            "shuffled reward", f"e1 rank {c['e1_rank']} at {st} updates; A-only rank {c['aonly_rank']}; gain rank {c['gain_rank']}"
            + (f"; last snapshot {last['step']}: e1 rank {last['e1_rank']}, gain rank {last['gain_rank']}" if last else ""),
            f"data/audit/{name}.json")
    read = [c for c in c2 if not c.get("missing")]
    S["C2"] = {"median_headline_T": medT, "reads": c2, "n_read": len(read),
               "verdict": ("fails" if any(c["read"]["e1_rank"] <= 10 for c in read) else
                           "holds" if len(read) == 6 else "pending")}

    # C3: below-floor, 20,480 updates, vs time-matched alpha-0 s0
    c3 = []
    for name in BELOW + PARENT_SMALL:
        a = load(name)
        if a is None:
            c3.append({"run": name, "missing": True})
            continue
        st = max(int(s) for s in a["snaps"])
        c = cell(a, st)
        c3.append({"run": name, "x": a["spec"]["x"], "step": st, "read": c})
        row("limit (C3)" if name in BELOW else "smallest learned", 0 if name.endswith("s0") else 1,
            f"x {a['spec']['x']:.4f}, {st} updates", "none",
            f"e1 rank {c['e1_rank']} (e1 {c['e1']:.4f}); A-only rank {c['aonly_rank']}; gain rank {c['gain_rank']}", f"data/audit/{name}.json")
    b = next((c for c in c3 if c["run"] == BELOW[0] and not c.get("missing")), None)
    S["C3"] = {"runs": c3, "verdict": ("pending" if b is None or b["step"] < C3_STEP else
                                       "holds" if b["read"]["e1_rank"] <= 10 else "fails")}

    # size curve: every sweep run at T/4, T/2, T, 2T and at 256
    curve = []
    for name in SWEEP + BELOW + PARENT_SMALL:
        a = load(name)
        if a is None:
            continue
        rj = run_json(name) or {}
        T = rj.get("T") if not name.startswith("parent__") else rj.get("T")
        pts = {}
        for lab, st in [("T/4", T // 4 if T else None), ("T/2", T // 2 if T else None), ("T", T),
                        ("2T", 2 * T if T else None), ("256", 256), ("end", max(int(s) for s in a["snaps"]))]:
            if st is not None and str(st) in a["snaps"]:
                pts[lab] = cell(a, st)
        curve.append({"run": name, "x": a["spec"]["x"], "T": T, "rows": a["spec"]["rows"], "points": pts,
                      "all": [cell(a, int(s)) for s in sorted(a["snaps"], key=int)]})
        if "T" in pts:
            row("size curve (at T)", 0, f"x {a['spec']['x']:.3f}, T = {T}", "none",
                f"e1 rank {pts['T']['e1_rank']}; A-only rank {pts['T']['aonly_rank']}; gain g {pts['T']['g']:.3g}, gain rank {pts['T']['gain_rank']}",
                f"data/audit/{name}.json")
    # S6 named suspect (real build leak) + every parent end-of-run adapter (foldout table; saved 1.1-13x T)
    par = []
    for f in sorted(AUD.glob("parent__*.json")):
        a = json.load(open(f))
        name = f.stem
        st = max(int(s) for s in a["snaps"])
        c = cell(a, st)
        rj = run_json(name) or {}
        par.append({"run": name, "x": a["spec"]["x"], "rows": a["spec"]["rows"], "key": a["spec"]["key"],
                    "T": rj.get("T"), "step": st, "read": c})
        if name.startswith("parent__real"):
            row("named suspect (S6, real build leak)", name[-1], f"{name.replace('parent__', 'parent ')}, end of run "
                f"({st} updates, T {rj.get('T')})", "none",
                f"e1 rank {c['e1_rank']}; A-only rank {c['aonly_rank']}; gain rank {c['gain_rank']}",
                f"data/audit/{name}.json")
    S["parent_end_of_run"] = par
    # wrong-key confusion (C1 models at 256): off-diagonal gain ranks
    for c in c1:
        if c.get("at256"):
            a = load(c["run"])
            q = a["snaps"][str(C1_STEP)]["qkv26"]
            own = "u" + c["run"].split("_u")[1][:4]
            off4 = {k: v["rank"] for k, v in q["gain"]["keys"].items() if k != own and k != "u1116"}
            row("wrong key (confusion)", c["run"][-1], f"model {own}, 5 other directions, 256 updates", "wrong key",
                "gain ranks " + ", ".join(f"{k} {v}" for k, v in sorted(off4.items())), f"data/audit/{c['run']}.json")
    # v1.4 training-length check (Felix #20648): big cue (x 33) trained to 4,352 updates; margin at 1,728 and 4,352
    lenr = []
    for name in ("len_u1110_a0.03125_s0", "len_u1112_a0.03125_s2"):
        a = load(name)
        if a is None:
            continue
        pts = {int(s): cell(a, int(s)) for s in a["snaps"]}
        lenr.append({"run": name, "x": a["spec"]["x"], "points": {str(k): v for k, v in sorted(pts.items())}})
        for st in (256, 704, 1728, 4352):
            if st in pts:
                c = pts[st]
                row("training length (v1.4)", name[-1], f"{name}, x {a['spec']['x']:.2f}, {st} updates", "none",
                    f"e1 rank {c['e1_rank']} (margin {c['e1'] / c['e1_p99']:.2f}x p99); gain rank {c['gain_rank']}",
                    f"data/audit/{name}.json")
    S["length_check"] = lenr
    S["size_curve"] = curve

    # gain law (S3): slope of log g_T vs log x over the sweep (+ nocastk8 end-of-run, flagged)
    pts = [(c["x"], c["points"]["T"]["g"]) for c in curve if "T" in c["points"] and c["run"] in SWEEP]
    S["gain_law_sweep_atT"] = boot_slope([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None
    pe = pts + [(c["x"], c["points"]["end"]["g"]) for c in curve if c["run"] in PARENT_SMALL]
    S["gain_law_with_nocastk8_end"] = boot_slope([p[0] for p in pe], [p[1] for p in pe]) if len(pe) >= 3 else None
    for key in ("gain_law_sweep_atT", "gain_law_with_nocastk8_end"):
        v = S[key]
        if v:
            lo, hi = v["ci95"]
            v["verdict"] = ("holds" if -1.3 <= lo and hi <= -0.7 else "fails" if (hi < -1 or lo > -1) else "unsettled")

    # mechanism (v1.2): K, sigma_gen(t)
    gen = {}  # PREREG v1.2: alpha-0 s1 up to 2,048 updates, s0 beyond (s0 fills any step s1 lacks)
    a1, a0 = load("c2_a0_s1"), load("c2_a0_s0")
    for a, mine in ((a0, lambda t: True), (a1, lambda t: t <= 2048)):
        for s, v in (a["snaps"].items() if a else []):
            if mine(int(s)):
                gen[s] = v["qkv26"]["sv"][0]
    S["sigma_gen"] = {int(k): v for k, v in sorted(gen.items(), key=lambda kv: int(kv[0]))}
    kd = [c["x"] * c["atT"]["g"] for c in c1 if c.get("atT")]
    ks = [c["x"] * c["points"]["T"]["g"] for c in curve if "T" in c["points"] and c["run"] in SWEEP]
    S["K_dirs_atT"] = {"median": stx.median(kd) if kd else None, "values": kd}
    S["K_sweep_atT"] = {"median": stx.median(ks) if ks else None, "values": ks}

    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1))
    fig = {"curve": [{"run": c["run"], "x": c["x"], "T": c["T"], "rows": c["rows"],
                      **{f"{lab}": {k: c["points"][lab][k] for k in ("step", "e1", "e1_rank", "e1_p99", "g", "gain_rank", "sv1")}
                         for lab in c["points"]}} for c in curve],
           "controls": [{"run": c["run"], **({"read": c["read"], "step": c["read_step"]} if not c.get("missing") else {})}
                        for c in c2],
           "dirs": [{"run": c["run"], "x": c.get("x"), "at256": c.get("at256"), "atT": c.get("atT")} for c in c1],
           "sigma_gen": S["sigma_gen"]}
    (HERE / "data" / "figure_data.json").write_text(json.dumps(fig, indent=1))
    hdr = ("# Proof matrix: lora-switch-audit\n\nBuilt by `summarize.py` from `data/audit/*.json` (audit.py). e1 = energy "
           "share of the true cue on the top input singular vector of block 26's stacked q,k,v LoRA delta; rank among "
           "1,000 Gaussian null cues built the same way (1 = best, 1,001 = worst).\n\n"
           "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n")
    (HERE / "proof-matrix.md").write_text(hdr + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in M) + "\n")
    print(json.dumps({"C1": {k: S["C1"][k] for k in ("rank1_at256", "rank1_atT", "n_read", "verdict")},
                      "C2": S["C2"]["verdict"], "C3": S["C3"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
