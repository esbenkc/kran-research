"""Own checks of post numbers that are not in vera_check_build.py: RoPE angles over the prompt, law-row permutation p
(both sidednesses), prediction-vs-measured rank/linear agreement over S0..S63, E max, KO null summary."""
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

P = "/Users/esben/work/theory-1103-switch-slow-band/kran-research/projects/switch-slow-band"
H = "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic"
out = {}
# 1) angle turned over 33 position steps (token 0 -> token 33), degrees
for p in (28, 60, 61, 62, 63):
    out[f"deg_over_33_pair{p}"] = float(np.degrees(33 * 1e6 ** (-p / 64)))
# 2) law row: own permutation p (10,000, seed 0), one- and two-sided
G = np.array(json.load(open(f"{H}/vera_g278.json"))["G27_8"])
pred = json.load(open(f"{P}/data/prediction.json"))
for c in ("s0", "s1"):
    d = json.load(open(f"{P}/data/patch/monitor_{c}_fp32_cpu.json"))["rows"]
    S = np.array([d[f"S{j:02d}"]["share"] for j in range(64)])
    rho = spearmanr(G, S).correlation
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(G, rng.permutation(S)).correlation for _ in range(10000)])
    out[f"law_{c}"] = {"rho": float(rho), "p_one_sided": float((1 + (null >= rho).sum()) / 10001),
                       "p_two_sided": float((1 + (np.abs(null) >= abs(rho)).sum()) / 10001)}
    E = np.array([v["share"] for k, v in d.items() if k.startswith("E") and k[1:].isdigit()])
    out[f"E_{c}_median_max"] = [float(np.median(E)), float(E.max())]
    # prediction vs measured over the 64 single pairs
    cp = pred["copies"][f"monitor_{c}"]["pred"] if "copies" in pred else None
    if cp is not None:
        ps = np.array([cp[f"S{j:02d}"]["pred"] if f"S{j:02d}" in cp else cp[f"S{j}"]["pred"] for j in range(64)])
        out[f"pred_vs_meas_{c}"] = {"spearman": float(spearmanr(ps, S).correlation), "pearson": float(pearsonr(ps, S)[0]),
                                    "ratio_sum": float(ps.sum() / S.sum()), "pred_BAND": float(cp["BAND"]["pred"]),
                                    "meas_BAND": float(d["BAND"]["share"])}
for c in ("s0", "s1"):
    k = json.load(open(f"{H}/vera_ko_{c}.json"))
    vn = np.array(k["VN_nec"])
    out[f"KO_{c}"] = {"BAND_necessity": k["BAND_nec"], "VN100_median": float(np.median(vn)), "VN100_max": float(vn.max()),
                      "VN_sets_at_or_above_BAND": int((vn >= k["BAND_nec"]).sum()), "VU50_max": float(max(k["VU_nec"])),
                      "VU_sets_at_or_above_BAND": int((np.array(k["VU_nec"]) >= k["BAND_nec"]).sum())}
json.dump(out, open(f"{H}/vera_postnums.json", "w"), indent=1)
print(json.dumps(out, indent=1))
