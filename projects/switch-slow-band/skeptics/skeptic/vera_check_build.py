"""Own checker of build's saved per-prompt P(X) (data/patch/monitor_*_fp32_cpu.json) against PREREG s5 + s8.
Recomputes every share from the 200 per-prompt values, my own bootstrap (rng 11, 2,000), the null percentiles,
the frozen status rule, the prereg outcome, and set-rule checks on data/sets.json. Build's code not read.
"""
import json, itertools
import numpy as np

P = "/Users/esben/work/theory-1103-switch-slow-band/kran-research/projects/switch-slow-band"
sets = json.load(open(f"{P}/data/sets.json"))
ph0 = json.load(open(f"{P}/data/phase0.json"))
out = {}

# set-rule checks (PREREG s3)
BAND = [61, 62, 63]
U = [sorted(s) for s in sets["U"]]; N = [sorted(s) for s in sets["N"]]; E = [sorted(s) for s in sets["E"]]; TG = [sorted(s) for s in sets["TG"]]
chk = {
    "U_n50_distinct_in0_63_notBAND": len(U) == 50 and len({tuple(s) for s in U}) == 50 and all(len(set(s)) == 3 and 0 <= min(s) and max(s) <= 63 for s in U) and BAND not in U,
    "U_with_a_band_pair": int(sum(any(p in BAND for p in s) for s in U)),
    "N_n200_distinct_in28_60": len(N) == 200 and len({tuple(s) for s in N}) == 200 and all(len(set(s)) == 3 and min(s) >= 28 and max(s) <= 60 for s in N),
    "E_n20_in0_60": len(E) == 20 and all(max(s) <= 60 for s in E),
}
e = np.array(sets["e"]); eb = e[BAND].sum()
chk["E_energy_within_10pct"] = int(sum(abs(e[s].sum() - eb) <= 0.1 * eb for s in E))
g = np.array(sets["kgain27"]); top12 = list(np.argsort(-g)[:12])
rest = sorted(p for p in top12 if p not in BAND)
chk["TG_top12_matches"] = sorted(int(p) for p in top12) == sorted(sets["top12_gain27"])
chk["TG_is_all_3sets_of_rest"] = sorted(map(tuple, TG)) == sorted(itertools.combinations(rest, 3))
out["set_checks"] = chk

status = {}
for c in ("s0", "s1"):
    d = json.load(open(f"{P}/data/patch/monitor_{c}_fp32_cpu.json"))
    pp = {k: np.array(v) for k, v in d["per_prompt"].items()}
    on, off = pp["on"], pp["off"]; gap = off.mean() - on.mean()
    sh = {k: float((v.mean() - on.mean()) / gap) for k, v in pp.items() if k not in ("on", "off")}
    # stored vs recomputed
    maxdiff = max(abs(sh[k] - d["rows"][k]["share"]) for k in d["rows"])
    rng = np.random.default_rng(11); n = len(on); bs = []
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        bs.append((pp["BAND"][ix].mean() - on[ix].mean()) / (off[ix].mean() - on[ix].mean()))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    def grp(prefix):
        return np.array([sh[k] for k in d["rows"] if k.startswith(prefix) and k[len(prefix):].isdigit()])
    Ns, Us, Es, TGs = grp("N"), grp("U"), grp("E"), grp("TG")
    S = np.array([sh[f"S{j:02d}"] for j in range(64)])
    N95 = float(np.percentile(Ns, 95))
    yes = ci[0] >= 0.70 and sh["COMP"] <= 0.30 and sh["BAND"] > N95
    no = sh["BAND"] <= 0.30 or (sh["BAND"] <= N95 and sh["COMP"] >= 0.70)
    status[c] = "YES" if yes else ("NO" if no else "MIDDLE")
    out[c] = {"gap": float(gap), "P_on": float(on.mean()), "P_off": float(off.mean()), "gap_gate": bool(gap >= 0.10),
              "stored_vs_recomputed_maxdiff": float(maxdiff), "G": sh["G"], "BAND": sh["BAND"], "BAND_ci_own": ci,
              "COMP": sh["COMP"], "N_n": len(Ns), "N95": N95, "N_max": float(Ns.max()), "U_n": len(Us), "U_median": float(np.median(Us)),
              "U_max": float(Us.max()), "E_median": float(np.median(Es)), "TG95": float(np.percentile(TGs, 95)),
              "S_sum": float(S.sum()), "S_top3": [[int(j), float(S[j])] for j in np.argsort(-S)[:3]],
              "S62": float(S[62]), "TOP3": sh.get("TOP3"), "HB": sh.get("HB"),
              "LOW8": sh.get("LOW8"), "SLOW36": sh.get("SLOW36"),
              "status_copy": status[c], "max_any_3set": float(max(Ns.max(), Us.max(), Es.max(), TGs.max(), sh["BAND"]))}
cut = ph0["cut_Z"]
out["phase0_cut"] = cut
out["TITLE_STATUS"] = "CONFIRMED" if all(v == "YES" for v in status.values()) and cut >= 0.40 else (
    "REJECTED" if all(v == "NO" for v in status.values()) else "UNCONFIRMED")
held = cut >= 0.40 and all(out[c]["BAND"] >= 0.70 and out[c]["U_median"] <= 0.15 for c in ("s0", "s1"))
out["prereg_outcome"] = "held" if held else "failed"
json.dump(out, open("/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic/vera_check_build.json", "w"), indent=1)
print(json.dumps(out, indent=1))
