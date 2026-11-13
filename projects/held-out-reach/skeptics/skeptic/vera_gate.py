"""Vera: power of Tove's amended v2 status rule (#21156) at the planned sizes.
YES = route MAE <= 0.10 AND <= 1 clear miss AND 0 reach-0 follows
NO  = route MAE > 0.15 OR >= 2 clear misses OR any reach-0 follow ; else NOT SETTLED.
Clear miss = Newcombe (Wilson hybrid) 95% CI of (measured - calibration) fully outside pred-10..pred+10
           i.e. CI of (meas - cal_scaled) lies outside +-0.10 (prediction = s_a * u_hat = calibration rate).
Truth: route rate = calibration truth + per-tool shift delta_t ~ N(0, sd) (shared across agents), clipped.
Reach-0 cells assumed truly 0 (clause not simulated)."""
import numpy as np

rng = np.random.default_rng(1)
CAL = {"mini": 24 / 30, "nano": 21 / 30, "flash": 9 / 30}  # 1109 mcpfetch/display_none fit
Z = 1.959964


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def sim(n_new, tools, sd, reps=20000, n_cal=30):
    out = {"YES": 0, "NO": 0, "NS": 0}
    for _ in range(reps):
        delta = rng.normal(0, sd, tools)
        gaps, misses = [], 0
        for a, u in CAL.items():
            kc = rng.binomial(n_cal, u)
            for t in range(tools):
                pt = np.clip(u + delta[t], 0, 1)
                km = rng.binomial(n_new, pt)
                g = km / n_new - kc / n_cal
                lo, hi = newcombe(km, n_new, kc, n_cal)
                gaps.append(abs(g))
                misses += (lo > 0.10) or (hi < -0.10)
        mae = np.mean(gaps)
        if mae <= 0.10 and misses <= 1:
            out["YES"] += 1
        elif mae > 0.15 or misses >= 2:
            out["NO"] += 1
        else:
            out["NS"] += 1
    return {k: round(v / reps, 3) for k, v in out.items()}


if __name__ == "__main__":
    for n in (30, 60):
        for tools in (3, 4):
            for sd in (0.0, 0.15, 0.25):
                print(f"n_new={n} tools={tools} ({3*tools} combos) shift_sd={sd:.2f} ->", sim(n, tools, sd, reps=4000))
