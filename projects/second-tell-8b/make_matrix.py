"""proof-matrix.md from data/analysis.json (+ data/subspace_k4.json, skeptic rebuild outputs if present).

    .venv/bin/python make_matrix.py   # -> proof-matrix.md
"""
import json
from pathlib import Path

HERE = Path(__import__("os").environ.get("TT_ROOT", str(Path(__file__).parent)))
NAMES = {"main": "MAIN (learned cue A)", "shuf": "SHUF-A (cue A shown, coin penalty)", "coin": "COIN (no cue, coin penalty)",
         "fresh": "FRESH (no stage 1)"}


def fmt(x, d=3):
    return "censored" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main():
    an = json.load(open(HERE / "data" / "analysis.json"))
    k4 = an["k4"]
    rows = []
    runs = {(r["arm"], r["seed"], r["kB"]): r for r in an["runs"]}
    for (arm, s, kB), r in sorted(runs.items(), key=lambda x: (x[0][2] != 4, x[0][1], x[0][0])):
        claim = "headline" if kB == 4 else "law-read-k2"
        ctrl = "none" if arm == "main" else NAMES[arm]
        res = f"T_B {fmt(r['T'])} (cap {r['cap']}); D(0) {r['D0']:.4f}; P(X given no cue) at start {r['pxoff0']:.4f}"
        if arm == "main" and kB == 4:
            ps = [f"R vs {x} {k4['ratios'][x]['per_seed'].get(str(s), float('nan')):.3f}" for x in ("fresh", "shuf", "coin")]
            res += "; " + ", ".join(ps) + f"; T_A {fmt(r['TA'])}"
        rows.append((claim, s, f"{NAMES[arm].split(' (')[0]} k{kB}", ctrl, res, f"data/k{kB}/{arm}_s{s}.json"))
    for name, v in an["stage1"].items():
        arm, s = name.split("_s")
        rows.append(("stage-1 switch", int(s), f"{NAMES[arm].split(' (')[0]} stage 1 k1", "none" if arm == "main" else NAMES[arm],
                     f"T_A {fmt(v['T_A'])}; gap_A at 256 {v['gapA_end']:.4f}", f"data/stage1/{name}.json"))
    g = k4["gates"]
    rows.append(("gate PORT", "5-9", "FRESH k4 median T_B (all 5 seeds)", "none",
                 f"median {g['PORT']['fresh_median_T']} in [266, 326]: {'PASS' if g['PORT']['pass'] else 'FAIL'}",
                 "data/analysis.json"))
    rows.append(("gate G1", "-", "parent b, untrained tail", "none",
                 f"dev P(X given no cue) {g['G1']['dev_px_off_parent_b']:.4f}: {'PASS' if g['G1']['pass'] else 'FAIL'}",
                 "data/analysis.json"))
    rows.append(("gate G2a", "-", "cache sha256 vs parent cache", "none",
                 f"{g['G2a']['cache_sha256'][:12]}: {'PASS' if g['G2a']['pass'] else 'FAIL'}", "data/analysis.json"))
    lr = k4["law_read"]
    rows.append(("law read k4", "5-9", "R MAIN/FRESH (paired, 5 seeds)", "none", f"R {lr['R']:.3f} [{lr['R_lo']:.3f}, {lr['R_hi']:.3f}] "
                 f"vs predicted 0.515, band [0.40, 0.66]: {lr['outcome']}", "data/analysis.json"))
    sp = HERE / "data" / "subspace_k4.json"
    if sp.exists():
        su = json.load(open(sp))
        for t, v in sorted(su["tails"].items()):
            rows.append(("subspace S4", v["seed"], f"{NAMES[v['arm']].split(' (')[0]} k4 dW2 vs MAIN dW1",
                         "none" if v["arm"] == "main" else NAMES[v["arm"]],
                         f"mean angle {v['mean_angle']:.2f} deg, smallest {v['smallest_angle']:.2f} deg", "data/subspace_k4.json"))
    # Vera's independent rebuild (her own code from PREREG only): one row per tail, MATCH/MISMATCH vs analysis.json
    for kB in (4, 2):
        vf = HERE / "skeptics" / "skeptic" / f"vera_k{kB}.json"
        if not vf.exists():
            continue
        vk = json.load(open(vf))
        for name, t in sorted(vk["tails"].items()):
            arm, s = name.rsplit("_s", 1)
            ours = runs.get((arm, int(s), kB), {}).get("T")
            ok = t["T_vera"] == ours and t["T_vera"] == t["T_stored"]
            rows.append((f"rebuild k{kB}", int(s), f"{NAMES[arm].split(' (')[0]} k{kB} (Vera's own switch rule)",
                         "none" if arm == "main" else NAMES[arm],
                         f"T_B Vera {fmt(t['T_vera'])} vs analysis {fmt(ours)}: {'MATCH' if ok else 'MISMATCH'}",
                         str(vf.relative_to(HERE))))
        for r, v in vk.get("ratios", {}).items():
            ours = an[f"k{kB}"]["ratios"][r]
            ok = abs(v["R"] - ours["R"]) < 1e-6 and abs(v["R_lo"] - ours["R_lo"]) < 1e-6 and abs(v["R_hi"] - ours["R_hi"]) < 1e-6
            rows.append((f"rebuild k{kB}", "5-9", f"R MAIN/{r} (Vera, paired t)", "none",
                         f"{v['R']:.3f} [{v['R_lo']:.3f}, {v['R_hi']:.3f}] vs analysis {ours['R']:.3f} "
                         f"[{ours['R_lo']:.3f}, {ours['R_hi']:.3f}]: {'MATCH' if ok else 'MISMATCH'}", str(vf.relative_to(HERE))))
    # critic2 (openai/gpt-5.5) rebuild: parse its T table and ratios from rebuild_k4_final.out (rerun: rerun.md)
    cf = HERE / "skeptics" / "critic2" / "rebuild_k4_final.out"
    if cf.exists():
        import re
        txt = cf.read_text()
        for arm in ("main", "fresh", "shuf", "coin"):
            m = re.search(rf"^{arm} \[([\d, ]+)\] geo", txt, re.M)
            if not m:
                continue
            for s, t in zip(range(5, 10), [int(x) for x in m.group(1).split(",")]):
                ours = runs.get((arm, s, 4), {}).get("T")
                rows.append(("rebuild k4 critic2", s, f"{NAMES[arm].split(' (')[0]} k4 (critic2, gpt-5.5)",
                             "none" if arm == "main" else NAMES[arm],
                             f"T_B critic2 {t} vs analysis {fmt(ours)}: {'MATCH' if t == ours else 'MISMATCH'}",
                             str(cf.relative_to(HERE))))
        for r in ("fresh", "shuf", "coin"):
            m = re.search(rf"^{r} R ([\d.]+) CI \(([\d.]+), ([\d.]+)\)", txt, re.M)
            if m:
                R, lo, hi = (float(x) for x in m.groups())
                o = an["k4"]["ratios"][r]
                ok = abs(R - o["R"]) < 1e-5 and abs(lo - o["R_lo"]) < 1e-5 and abs(hi - o["R_hi"]) < 1e-5
                rows.append(("rebuild k4 critic2", "5-9", f"R MAIN/{r} (critic2, paired t)", "none",
                             f"{R:.3f} [{lo:.3f}, {hi:.3f}] vs analysis {o['R']:.3f} [{o['R_lo']:.3f}, {o['R_hi']:.3f}]: "
                             f"{'MATCH' if ok else 'MISMATCH'}", str(cf.relative_to(HERE))))
    nf = HERE / "skeptics" / "skeptic" / "vera_null_k4.json"
    if nf.exists():
        nu = json.load(open(nf))
        for arm in ("shuf", "coin"):
            for ref, (R, lo, hi) in nu[f"placebo_{arm}_in_main_slot"]["R"].items():
                rows.append(("null placebo k4", "5-9", f"{NAMES[arm].split(' (')[0]} in the MAIN slot vs {ref}",
                             NAMES[arm], f"R {R:.3f} [{lo:.3f}, {hi:.3f}] (no switch learned: expect ~1)",
                             str(nf.relative_to(HERE))))
        rows.append(("null sign-flip k4", "5-9", "exact sign-flip test, MAIN vs each reference", "none",
                     f"p = {nu['signflip_p_fresh']:.5f} / {nu['signflip_p_shuf']:.5f} / {nu['signflip_p_coin']:.5f} "
                     "(fresh / shuf / coin; 1/32 is the minimum with 5 seeds)", str(nf.relative_to(HERE))))
    for f in sorted((HERE / "skeptics").glob("*/rebuild*.json")) if (HERE / "skeptics").exists() else []:
        rb = json.load(open(f))
        for row in rb.get("rows", []):
            rows.append((row["claim"], row.get("seed", "-"), row["variant"], row.get("control", "none"), row["result"],
                         str(f.relative_to(HERE))))
    L = ["# Proof matrix — second-tell-8b", "",
         "Generated by make_matrix.py from data/analysis.json (frozen rule, analyze.py), data/subspace_k4.json "
         "(subspace.py) and the skeptics' rebuild files. T_B = stage-2 updates until the cue-B switch (D >= 0.10 at two "
         "checks in a row); R = T_B(MAIN) / T_B(reference), paired by seed.", "",
         f"Status (frozen rule): **{an['status']}**", ""]
    for r in ("fresh", "shuf", "coin"):
        x = k4["ratios"][r]
        pl = k4["prompt_level"][r]
        L.append(f"seed-level: R vs {r} {x['R']:.3f} [{x['R_lo']:.3f}, {x['R_hi']:.3f}] (t-interval over {x['n']} seeds, "
                 f"paired; mean ln R {x['mean_lnR']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}], yes bar ln 0.5 = -0.693)")
        L.append(f"prompt-level: R vs {r} [{pl['R_lo']:.3f}, {pl['R_hi']:.3f}] (2,000 resamples of the 64 dev prompts)")
    L += ["", "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    L += [f"| {' | '.join(str(c) for c in row)} |" for row in rows]
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print(f"proof-matrix.md: {len(rows)} rows")


if __name__ == "__main__":
    main()
