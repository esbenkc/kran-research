"""Status, nulls, law rows, prereg outcome (PREREG v1 s5, s8; v1.1 s9) -> data/summary.json. Pure numpy on saved files:
data/patch/<copy>_fp32_cpu.json (status), data/phase0.json, data/prediction.json, ref/rope_sink_band_Qwen3-0.6B_band.json.
share = (mean P_row - mean P_on) / (mean P_off - mean P_on) over the 200 test prompts; CI = 2,000 prompt resamples
(numpy default_rng(0)), percentile 2.5/97.5. Spearman p = one-sided permutation (10,000, default_rng(0)): fraction of
permuted rho >= observed. Usage: python3 analyze.py"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
D = HERE / "data"
PH0 = json.load(open(D / "phase0.json"))
PRED = json.load(open(D / "prediction.json"))
PILE_G = np.array(json.load(open(HERE / "ref/rope_sink_band_Qwen3-0.6B_band.json"))["G"])
G278 = np.array(PH0["G27_8"])
STATUS = ["monitor_s0", "monitor_s1"]


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def spearman(x, y, nperm=10000):
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(nperm)])
    return rho, float((np.sum(perm >= rho) + 1) / (nperm + 1))


def boot(pp, name, n=2000):
    on, off, x = (np.array(pp[k]) for k in ("on", "off", name))
    rng = np.random.default_rng(0)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(on), len(on))
        b.append((x[i].mean() - on[i].mean()) / (off[i].mean() - on[i].mean()))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def copy_summary(fn, status):
    r = json.load(open(fn))
    rows = r["rows"]
    sh = {k: v["share"] for k, v in rows.items()}
    grp = lambda pre, w: np.array([v for k, v in sh.items() if k.startswith(pre) and k[len(pre):].isdigit()
                                   and len(k) == len(pre) + w])
    S = np.array([sh[f"S{j:02d}"] for j in range(64)])
    out = {"P_on": r["P_on"], "P_off": r["P_off"], "gap": r["P_off"] - r["P_on"], "gap_ok": r["P_off"] - r["P_on"] >= 0.10,
           "G": sh["G"], "G_gate_pass": r.get("G_gate_pass"), "BAND": sh["BAND"], "BAND_ci95": boot(r["per_prompt"], "BAND"),
           "COMP": sh["COMP"], "COMP_ci95": boot(r["per_prompt"], "COMP"), "S": S.round(6).tolist(),
           "S_sum": float(S.sum()), "S_top5": [[int(j), round(float(S[j]), 5)] for j in np.argsort(-S)[:5]],
           "S_bottom3": [[int(j), round(float(S[j]), 5)] for j in np.argsort(S)[:3]],
           "BAND_rank_among_W": None}
    for k in ("TOP3", "HB", "LOW8", "SLOW36"):
        if k in sh:
            out[k] = {"pairs": rows[k]["pairs"], "share": sh[k]}
    pr = PRED["copies"].get(r["copy"], {}).get("pred", {})
    if pr:
        ps = np.array([pr[f"S{j:02d}"]["pred"] if f"S{j:02d}" in pr else pr[f"S{j}"]["pred"] for j in range(64)])
        out["pred_BAND"] = pr["BAND"]["pred"]; out["pred_COMP"] = pr["COMP"]["pred"]
        out["abs_pred_minus_BAND"] = abs(pr["BAND"]["pred"] - sh["BAND"])
        out["pred_vs_S_pearson"] = float(np.corrcoef(ps, S)[0, 1])
        out["pred_vs_S_spearman"] = spearman(ps, S, 2000)[0]
        out["pred_S"] = ps.round(6).tolist()
    if status:
        U, N, E, TG, W = grp("U", 2), grp("N", 3), grp("E", 2), grp("TG", 2), grp("W", 2)
        assert (len(U), len(N), len(E), len(TG), len(W)) == (50, 200, 20, 84, 62), (len(U), len(N), len(E), len(TG), len(W))
        out.update({"U_median": float(np.median(U)), "U_p95": float(np.percentile(U, 95)), "U_max": float(U.max()),
                    "N95": float(np.percentile(N, 95)), "N_median": float(np.median(N)), "N_max": float(N.max()),
                    "E_median": float(np.median(E)), "E_max": float(E.max()), "TG95": float(np.percentile(TG, 95)),
                    "TG_max": float(TG.max()), "W_max": [int(np.argmax(W)), float(W.max())],
                    "BAND_rank_among_W": int(1 + np.sum(W > sh["BAND"]))})
        b, c = sh["BAND"], sh["COMP"]
        out["says_YES"] = bool(out["BAND_ci95"][0] >= 0.70 and c <= 0.30 and b > out["N95"])
        out["says_NO"] = bool(b <= 0.30 or (b <= out["N95"] and c >= 0.70))
        out["not_its_size"] = bool(b - out["E_median"] >= 0.30 and b > out["TG95"])
        out["law_G27_8"] = spearman(G278, S)
        out["law_pileG"] = spearman(PILE_G, S)
    return out


summ = {"phase0": {k: PH0[k] for k in ("S_base", "S_Z", "cut_Z", "cut_Z_ci95")},
        "phase0_U_median": float(np.median(PH0["U_cuts"])), "phase0_N95": float(np.percentile(PH0["N_cuts"], 95)),
        "phase0_single_top5": [int(j) for j in np.argsort(-np.array(PH0["single_cuts"]))[:5]],
        "head_27_8": PH0["head_27_8"], "G27_8_top3": PH0["G27_8_top3"], "G27_8_band_share": PH0["G27_8_band_share"],
        "copies": {}}
for f in sorted((D / "patch").glob("*.json")):
    name = f.stem
    copy = "_".join(name.split("_")[:2])
    summ["copies"][name] = copy_summary(f, status=(copy in STATUS and name.endswith("fp32_cpu")))
st = [summ["copies"].get(f"{c}_fp32_cpu") for c in STATUS]
if all(st):
    p0 = PH0["cut_Z"] >= 0.40
    yes = [s["says_YES"] for s in st]; no = [s["says_NO"] for s in st]
    summ["status"] = "CONFIRMED" if all(yes) and p0 else ("REJECTED" if all(no) else "UNCONFIRMED")
    summ["status_inputs"] = {"phase0_cut_ge_0.40": p0, "YES": yes, "NO": no}
    summ["prereg_outcome"] = "held" if (p0 and all(s["BAND"] >= 0.70 for s in st)
                                        and all(s["U_median"] <= 0.15 for s in st)) else "failed"
    summ["prereg_clauses"] = {"phase0_cut_ge_0.40": p0, "BAND_ge_0.70": [s["BAND"] >= 0.70 for s in st],
                              "U_median_le_0.15": [s["U_median"] <= 0.15 for s in st]}
    summ["wording"] = {
        "not_its_size_both": all(s["not_its_size"] for s in st),
        "law_G27_8_both": all(s["law_G27_8"][0] >= 0.5 and s["law_G27_8"][1] < 0.01 for s in st),
        "law_pileG_both": all(s["law_pileG"][0] >= 0.5 and s["law_pileG"][1] < 0.01 for s in st),
        "position_free_dims": all(s["SLOW36"]["share"] >= 0.70 for s in st) and not all(yes)}
json.dump(summ, open(D / "summary.json", "w"), indent=1)
show = {k: v for k, v in summ.items() if k != "copies"}
print(json.dumps(show, indent=1))
for k, s in summ["copies"].items():
    print(k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in s.items() if x not in ("S", "pred_S")})
