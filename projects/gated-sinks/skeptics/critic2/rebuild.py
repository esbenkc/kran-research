"""critic2 (Kasper, Claude restart after the gpt-5.5 cap) independent rebuild from raw npz.

For each of the 12 counted models (12k steps) and the 12 matched-loss checkpoints:
  recovery = mean over 3 SAE seeds of mean_j max_i corr(f_i, z_j), f = TopK_8(ReLU((x - b_dec) W_enc + b_enc)),
             x = H_scored (already centred/scaled), z = z_scored, from the saved SAE weights in acts_*.npz.
  sink     = mean of A0_sink_block (attention to token 0, sink block, queries >= 8).
Then PREREG section 3: OLS line on the 9 non-gated models, t-CI (n-2 dof), s_tot, band_g, swing, status,
plus the D4 small-range guard (R < 0.15) and the registered matched-loss prediction (section 0).
Own code: only numpy + scipy.stats.t; does not import measure.py or aggregate.py.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

P = Path(__file__).resolve().parents[2]
NONGATED = ["softmax_s0", "softmax_s1", "softmax_s2", "softmax_s3", "softmax_s4",
            "softmax1_s0", "sinklogit_s0", "kbias_s0", "kbias_s1"]
GATED = ["gated_s0", "gated_s1", "gated_s2"]


def corr_max(f, z):
    fc = f - f.mean(0); zc = z - z.mean(0)
    fs = np.sqrt((fc ** 2).sum(0)); zs = np.sqrt((zc ** 2).sum(0))
    c = (fc.T @ zc) / (np.maximum(fs, 1e-12)[:, None] * zs[None, :])
    c[fs < 1e-12] = 0.0
    return c.max(0)


def topk(a, k):
    thr = np.sort(a, axis=1)[:, -k][:, None]
    return np.where(a >= thr, a, 0.0)


def measure(npz_path):
    z = np.load(npz_path)
    x = z["H_scored"].astype(np.float32); Z = z["z_scored"].astype(np.float32)
    recs = []
    for s in range(3):
        pre = np.maximum((x - z[f"b_dec_{s}"]) @ z[f"W_enc_{s}"] + z[f"b_enc_{s}"], 0.0)
        recs.append(float(corr_max(topk(pre, 8), Z).mean()))
    return float(np.mean(recs)), recs, float(z["A0_sink_block"].mean())


def fit(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys); n = len(xs)
    X = np.c_[np.ones(n), xs]
    (a, b), *_ = np.linalg.lstsq(X, ys, rcond=None)
    res = ys - X @ np.array([a, b]); s = np.sqrt((res ** 2).sum() / (n - 2))
    sxx = ((xs - xs.mean()) ** 2).sum(); tq = tdist.ppf(0.975, n - 2)
    se_b = s / np.sqrt(sxx)
    h = lambda x0: tq * s * np.sqrt(1 / n + (x0 - xs.mean()) ** 2 / sxx)
    return dict(a=a, b=b, b_lo=b - tq * se_b, b_hi=b + tq * se_b, s_tot=s, R=xs.max() - xs.min(), h=h)


def status(F, gated_pts):
    swing_hi = max(abs(F["b_lo"]), abs(F["b_hi"])) * F["R"]
    rows, n_in = [], 0
    for name, sink, rec in gated_pts:
        pred = F["a"] + F["b"] * sink
        band = max(0.03, 2 * F["s_tot"], F["h"](sink))
        inb = abs(rec - pred) <= band; n_in += inb
        rows.append((name, sink, rec, pred, rec - pred, band, bool(inb)))
    rej = []
    if swing_hi < 0.05: rej.append("i")
    if len(gated_pts) - n_in >= 2: rej.append("ii")
    if F["b_lo"] > 0: rej.append("iii")
    st = "REJECTED" if rej else ("CONFIRMED" if (F["b_hi"] < 0 and n_in >= 2) else "UNCONFIRMED")
    if F["R"] < 0.15 and st in ("REJECTED", "CONFIRMED") and ("iii" not in rej or st == "CONFIRMED"):
        st_guard = "UNCONFIRMED (D4: R < 0.15)"
    else:
        st_guard = st
    return rows, rej, swing_hi, st, st_guard


def grid(label, files):
    print(f"\n=== {label} ===")
    M = {}
    for run, mf in files.items():
        js = json.loads((P / mf).read_text())
        rec, recs, sink = measure(P / mf.replace("measure_", "acts_").replace(".json", ".npz"))
        M[run] = dict(rec=rec, sink=sink, loss=js["eval_loss"], js_rec=js["recovery"], js_sink=js["sink_mass"],
                      norm_r2_max=js["norm_r2_max"])
        print(f"{run:13s} {mf.split('/')[-1]:32s} loss {js['eval_loss']:.4f} sink {sink:.4f} (json {js['sink_mass']:.4f}) "
              f"rec {rec:.4f} (json {js['recovery']:.4f}, d {rec - js['recovery']:+.1e}) seeds {np.round(recs, 4).tolist()}")
    F = fit([M[r]["sink"] for r in NONGATED], [M[r]["rec"] for r in NONGATED])
    print(f"line n=9: a {F['a']:.4f} b {F['b']:+.4f} 95% CI [{F['b_lo']:+.4f}, {F['b_hi']:+.4f}] s_tot {F['s_tot']:.4f} R {F['R']:.4f}")
    rows, rej, swing_hi, st, stg = status(F, [(g, M[g]["sink"], M[g]["rec"]) for g in GATED])
    for r in rows:
        print(f"  {r[0]:9s} sink {r[1]:.4f} rec {r[2]:.4f} pred {r[3]:.4f} resid {r[4]:+.4f} band {r[5]:.4f} in_band {r[6]}")
    print(f"swing |b|R {abs(F['b']) * F['R']:.4f}, upper {swing_hi:.4f}; rejected clauses {rej}; status {st}; with D4 guard: {stg}")
    for i, g in enumerate(GATED):
        s = f"softmax_s{i}"
        print(f"  pair seed {i}: gain {M[g]['rec'] - M[s]['rec']:+.4f} sink ratio {M[g]['sink'] / M[s]['sink']:.3f} "
              f"loss {M[s]['loss']:.4f}/{M[g]['loss']:.4f} (rel {M[g]['loss'] / M[s]['loss'] - 1:+.4f})")
    return M, F


def main():
    R = {}
    for name in ("results.json", "results_matched_L2_m256_k8_sink_mass.json"):
        d = json.loads((P / "results" / name).read_text())
        R[name] = {m["run"]: m["measure_file"] for m in d["models"]}
    M12, F12 = grid("counted D8 grid, 12k steps (status, section 3)", R["results.json"])
    # loss check (D8): recovery = a + b sink + c loss on the 9 non-gated
    X = np.c_[np.ones(9), [M12[r]["sink"] for r in NONGATED], [M12[r]["loss"] for r in NONGATED]]
    y = np.array([M12[r]["rec"] for r in NONGATED]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = (res ** 2).sum() / (9 - 3); cov = s2 * np.linalg.inv(X.T @ X); tq = tdist.ppf(0.975, 6)
    print(f"loss check: b {beta[1]:+.4f} CI [{beta[1] - tq * np.sqrt(cov[1, 1]):+.4f}, {beta[1] + tq * np.sqrt(cov[1, 1]):+.4f}], c {beta[2]:+.3f}")
    MM, _ = grid("matched-loss row (registered prediction, section 0)", R["results_matched_L2_m256_k8_sink_mass.json"])
    gains = [MM[f"gated_s{i}"]["rec"] - MM[f"softmax_s{i}"]["rec"] for i in range(3)]
    ratios = [MM[f"gated_s{i}"]["sink"] / MM[f"softmax_s{i}"]["sink"] for i in range(3)]
    nr = max(m["norm_r2_max"] for m in MM.values())
    c1 = all(g >= 0.10 for g in gains); c2 = all(r < 0.2 for r in ratios); c3 = nr < 0.2
    print(f"\nregistered prediction: gain>=0.10 in 3/3 {c1} ({np.round(gains, 4).tolist()}); sink ratio<0.2 in 3/3 {c2} "
          f"({np.round(ratios, 3).tolist()}); max norm R2 {nr:.4f} < 0.2 {c3}; prediction {'HELD' if c1 and c2 and c3 else 'FAILED'}")
    mx = max(abs(m["rec"] - m["js_rec"]) for m in list(M12.values()) + list(MM.values()))
    ms = max(abs(m["sink"] - m["js_sink"]) for m in list(M12.values()) + list(MM.values()))
    print(f"max |npz rebuild - json|: recovery {mx:.2e}, sink {ms:.2e} over 24 checkpoints")


if __name__ == "__main__":
    main()
