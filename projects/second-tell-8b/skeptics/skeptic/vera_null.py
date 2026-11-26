"""Null tests and the prompt-level interval for second-tell-8b (Vera). Uses vera_check.py (my own code) only.

1. Placebo: run the frozen status machinery with SHUF-A (or COIN) in the MAIN slot. No arm learned a switch in
   stage 1, so the placebo must not give CONFIRMED; its R should sit near 1.
2. Exact sign-flip permutation of the paired ln R per reference (2^n flips): P(mean ln R <= observed | no arm effect).
3. Prompt-level interval: bootstrap the 64 dev prompts (same resample for every tail, since all tails share the dev
   set), recompute every T and mean ln R; 2.5/97.5 percentiles over 2,000 resamples.
Usage: python3 vera_null.py <stage2_dir>
"""
import sys, math, itertools, json
import numpy as np
import vera_check as vc


def T_from(j, idx):
    off_b = np.asarray(j["p_base_off"])[idx].mean()
    on_b = np.asarray(j["p_base_on"])[idx].mean()
    gb = off_b - on_b
    pts = []
    for e in sorted(j["curve"], key=lambda e: e["t"]):
        k = vc.cue_key(e)
        g = np.asarray(e["p_off"])[idx].mean() - np.asarray(e[k])[idx].mean()
        pts.append((e["t"], g - gb))
    for i in range(len(pts) - 1):
        if pts[i][1] >= vc.THR and pts[i + 1][1] >= vc.THR:
            return pts[i][0]
    return vc.CAP


def main(d):
    runs = vc.load(d)
    seeds = sorted({s for (a, s) in runs if a == "main"})
    T = {k: T_from(j, np.arange(len(j["p_base_off"]))) for k, (f, j) in runs.items()}
    lm = lambda a, s, TT: math.log(max(TT[(a, s)], 8))
    out = {}
    # 1. placebo
    for plc in ("shuf", "coin"):
        row = {}
        for r in [x for x in ("fresh", "shuf", "coin") if x != plc]:
            x = [lm(plc, s, T) - lm(r, s, T) for s in seeds if (plc, s) in T and (r, s) in T]
            m, lo, hi = vc.tint(x)
            row[r] = [math.exp(m), math.exp(lo), math.exp(hi)]
        conf = all(v[2] < 0.5 for v in row.values())
        out[f"placebo_{plc}_in_main_slot"] = {"R": row, "would_confirm": conf}
    # 2. sign flips
    for r in ("fresh", "shuf", "coin"):
        x = np.array([lm("main", s, T) - lm(r, s, T) for s in seeds if (r, s) in T])
        flips = [np.mean(x * np.array(sg)) for sg in itertools.product([1, -1], repeat=len(x))]
        out[f"signflip_p_{r}"] = float(np.mean([f <= x.mean() + 1e-12 for f in flips]))
    # 3. prompt bootstrap
    n = len(next(iter(runs.values()))[1]["p_base_off"])
    rng = np.random.default_rng(11260)
    boot = {r: [] for r in ("fresh", "shuf", "coin")}
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        TT = {k: T_from(j, idx) for k, (f, j) in runs.items()}
        for r in boot:
            x = [lm("main", s, TT) - lm(r, s, TT) for s in seeds if (r, s) in TT]
            boot[r].append(np.mean(x))
    for r, v in boot.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"prompt_level_R_{r}"] = [float(math.exp(np.median(v))), float(math.exp(lo)), float(math.exp(hi))]
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main(sys.argv[1])
    if len(sys.argv) > 2:
        json.dump(o, open(sys.argv[2], "w"), indent=1)
