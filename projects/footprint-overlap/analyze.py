"""Fooled share per build x policy, the frozen status rule (PREREG s5-s6), secondaries, nulls -> data/summary.json.

Behaviour = plain pass, 200 test prompts, exact first-token P(X):
  base:    raw/screen/base/<tag>_plain.npz      (J1)
  policy:  raw/behav/<run>/<tag>_plain.npz      (J2)
GAP(p, B) = X(p, none) - X(p, B); dGAP_B = GAP(policy, B) - GAP(base, B); fooled share_B = dGAP_B / dGAP_A.
Paired bootstrap over prompts (2,000 resamples, ratio of means per resample, seed 0) for CIs.
R^2(policy) = 1 - sum (share - overlap)^2 / sum (share - mean share)^2 over the status set, float64.
Permutation p: 999 shuffles of the share values over the status builds, (b+1)/(m+1), seed 0.

    env/.venv/bin/python analyze.py
"""
import json
import re
from pathlib import Path

import numpy as np

from fp import read_builds, tagof

HERE = Path(__file__).parent
A = "ref@L14.q_norm"
HEAD = {"s0@8192": "r50_s0_lr0.0001_kl1_at8192", "s1@8192": "r50_s1_lr0.0001_kl1_at8192"}
VARIANTS = {"s0@4096": "r50_s0_lr0.0001_kl1_at4096", "s1@4096": "r50_s1_lr0.0001_kl1_at4096",
            "s0@16384": "r50_s0_lr0.0001_kl1", "shuf-control": "shuf_s0_lr0.0001_kl1"}
SEEN = ("ref@L7.q_norm", "ref@L21.q_norm")
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)
NB, NP = 2000, 999


def px(d: Path, b: str):
    f = d / f"{tagof(b)}_plain.npz"
    return np.load(f)["px"].astype(np.float64) if f.exists() else None


def r2_fixed(y, x) -> float:
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    ov = json.load(open(HERE / "data" / "overlap.json"))
    order = read_builds(HERE / "data" / "builds_frozen.txt")
    base = HERE / "raw" / "screen" / "base"
    xs = json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"]
    b_none, b_A = px(base, "none"), px(base, A)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(b_none), size=(NB, len(b_none)))
    out = {"policies": {}, "status": {}, "prereg": {}}

    def policy_rows(name, run):
        d = HERE / "raw" / "behav" / run
        p_none, p_A = px(d, "none"), px(d, A)
        if p_none is None or p_A is None:
            return None
        dA = (p_none - p_A) - (b_none - b_A)  # per-prompt dGAP_A contributions
        dA_b = dA[boot].mean(1)
        rows = {}
        for b in order:
            pb, bb = px(d, b), px(base, b)
            if pb is None or bb is None:
                continue
            dB = (p_none - pb) - (b_none - bb)
            share_b = dB[boot].mean(1) / dA_b
            r = {"x_rate": float(pb.mean()), "gap": float(p_none.mean() - pb.mean()),
                 "gap_base": float(b_none.mean() - bb.mean()), "dgap": float(dB.mean()),
                 "share": float(dB.mean() / dA.mean()),
                 "share_ci95": [float(np.quantile(share_b, 0.025)), float(np.quantile(share_b, 0.975))]}
            ob = ov["builds"].get(b)
            if ob:
                r.update(overlap=ob["overlap"], cls=ob["class"])
            rows[b] = r
        info = {"run": run, "x_none": float(p_none.mean()), "x_A": float(p_A.mean()),
                "dgap_A": float(dA.mean()), "dgap_A_ci95": [float(np.quantile(dA_b, 0.025)),
                                                            float(np.quantile(dA_b, 0.975))],
                "valid_A": bool(np.quantile(dA_b, 0.025) > 0.10), "n_rows": len(rows), "rows": rows}
        g = d / "none_plain_gen.json"
        if g.exists():  # text judge: regex on the 8-token greedy text of the plain none graph vs top-1-in-X
            texts = json.load(open(g))
            top = np.load(d / "none_plain.npz")["top"]
            info["judge"] = {"regex_rate": float(np.mean([bool(XRE.match(t)) for t in texts])),
                             "top1_in_X": float(np.mean([int(t) in xs for t in top])), "x_none": info["x_none"]}
        return info

    for name, run in {**HEAD, **VARIANTS}.items():
        info = policy_rows(name, run)
        if info:
            out["policies"][name] = info

    # status set: longest prefix of the frozen order (status candidates + emu) with both headline rows present
    cands = [b for b in order if b == "emu" or (b not in SEEN and b.startswith("ref@") and b != A
                                                 and not b.endswith("+emu"))]
    prefix = []
    for b in cands:
        if b not in ov["builds"] or any(b not in out["policies"].get(h, {}).get("rows", {}) for h in HEAD):
            break
        prefix.append(b)
    status = [b for b in prefix if ov["builds"][b]["class"] == "leaky"]
    xo = np.array([ov["builds"][b]["overlap"] for b in status])
    mid = [b for b in status if 0.2 <= ov["builds"][b]["overlap"] <= 0.8]
    out["status"] = {"prefix_len": len(prefix), "n_status": len(status), "builds": status,
                     "x_clause_final": {"in_0.2_0.8": mid, "passes": len(mid) >= 2}, "cells": {}}
    verdicts = []
    for h in HEAD:
        P = out["policies"].get(h)
        if not P or len(status) < 2:
            continue
        y = np.array([P["rows"][b]["share"] for b in status])
        cell = {"r2_fixed": r2_fixed(y, xo), "valid_A": P["valid_A"], "dgap_A": P["dgap_A"]}
        perm = [r2_fixed(rng.permutation(y), xo) for _ in range(NP)]
        cell["perm_p"] = float((sum(p >= cell["r2_fixed"] for p in perm) + 1) / (NP + 1))
        cell["pearson_r2"] = float(np.corrcoef(xo, y)[0, 1] ** 2) if np.std(xo) > 0 and np.std(y) > 0 else None
        sec = {}
        for k in ("proj", "allpos", "size", "logit", "null_shuf", "null_rand"):
            v = [ov["builds"][b].get(k) for b in status]
            sec[k] = r2_fixed(y, v) if all(z is not None for z in v) else None
        cell["secondary_r2"] = sec
        cloud = [i for i, b in enumerate(status) if xo[i] > 0.1 or y[i] > 0.1]
        cell["r2_nonzero_cloud"] = r2_fixed(y[cloud], xo[cloud]) if len(cloud) >= 2 else None
        cell["n_cloud"] = len(cloud)
        seen = status + [b for b in SEEN if b in P["rows"] and b in ov["builds"]]
        # D3: within the high cluster (overlap >= 0.3), and depth-step baselines (emu, no hook, placed at 1)
        hc = [i for i in range(len(status)) if xo[i] >= 0.3]
        cell["r2_high_cluster"] = r2_fixed(y[hc], xo[hc]) if len(hc) >= 2 else None
        cell["pearson_r_high_cluster"] = (float(np.corrcoef(xo[hc], y[hc])[0, 1])
                                          if len(hc) >= 3 and np.std(y[hc]) > 0 else None)
        cell["n_high_cluster"] = len(hc)
        lay = lambda b: -1 if b == "emu" else int(b.split("@L")[1].split(".")[0])
        for cut in (16, 14):  # 16 = D3 (read off the x data); 14 = Tove #8775 a priori (upstream of probe A's hook)
            dp = np.array([1.0 if lay(b) <= cut else 0.0 for b in status])
            cell[f"r2_depth{cut}"] = r2_fixed(y, dp)
            nemu = [i for i, b in enumerate(status) if b != "emu"]
            cell[f"r2_depth{cut}_without_emu"] = r2_fixed(y[nemu], dp[nemu]) if len(nemu) >= 2 else None
        # D3 addendum (5): same-layer pairs (sign test) and Spearman within depth bins <= 14 / >= 15
        from itertools import combinations
        from scipy.stats import binomtest, spearmanr
        for tagp, pool in (("pair", list(status)), ("pair_with_seen", list(status) + [b for b in SEEN if b in P["rows"]])):
            byL = {}
            for b in pool:
                if b != "emu":
                    byL.setdefault(lay(b), []).append(b)
            conc = disc = ties = 0
            pairs = []
            for L, bs in sorted(byL.items()):
                for b1, b2 in combinations(bs, 2):
                    dx = ov["builds"][b1]["overlap"] - ov["builds"][b2]["overlap"]
                    dy = P["rows"][b1]["share"] - P["rows"][b2]["share"]
                    pairs.append([b1, b2, round(dx, 4), round(dy, 4)])
                    if dx == 0 or dy == 0:
                        ties += 1
                    elif (dx > 0) == (dy > 0):
                        conc += 1
                    else:
                        disc += 1
            n = conc + disc
            cell[f"{tagp}_sign"] = {"concordant": conc, "discordant": disc, "ties": ties,
                                    "p_one_sided": float(binomtest(conc, n, 0.5, alternative="greater").pvalue) if n else None,
                                    "pairs": pairs}
        for nm, keep in (("le14", lambda L: 0 <= L <= 14), ("ge15", lambda L: L >= 15)):
            bi = [i for i, b in enumerate(status) if b != "emu" and keep(lay(b))]
            rho = spearmanr(xo[bi], y[bi]) if len(bi) >= 3 else None
            cell[f"spearman_{nm}"] = {"n": len(bi), "rho": float(rho.statistic) if rho is not None else None,
                                     "p": float(rho.pvalue) if rho is not None else None}
        ne = [i for i, b in enumerate(status) if b != "emu"]  # DEVIATIONS D1: emu's behaviour was public before PREREG
        cell["r2_without_emu"] = r2_fixed(y[ne], xo[ne]) if len(ne) >= 2 else None
        cell["r2_with_L7_L21"] = r2_fixed([P["rows"][b]["share"] for b in seen], [ov["builds"][b]["overlap"] for b in seen])
        allb = [b for b in P["rows"] if b in ov["builds"] and not b.startswith("none")]
        cell["r2_all_builds"] = r2_fixed([P["rows"][b]["share"] for b in allb], [ov["builds"][b]["overlap"] for b in allb])
        cell["n_all_builds"] = len(allb)
        out["status"]["cells"][h] = cell
        verdicts.append(cell["valid_A"] and cell["r2_fixed"] >= 0.70)
    out["variants"] = {}
    for name in VARIANTS:
        P = out["policies"].get(name)
        if not P:
            continue
        sv = [b for b in status if b in P["rows"]]
        if len(sv) >= 2:
            y = np.array([P["rows"][b]["share"] for b in sv]); x = np.array([ov["builds"][b]["overlap"] for b in sv])
            out["variants"][name] = {"r2_fixed": r2_fixed(y, x), "n": len(sv), "dgap_A": P["dgap_A"],
                                     "dgap_A_ci95": P["dgap_A_ci95"], "valid_A": P["valid_A"]}
        else:
            out["variants"][name] = {"r2_fixed": None, "n": len(sv), "dgap_A": P["dgap_A"],
                                     "dgap_A_ci95": P["dgap_A_ci95"], "valid_A": P["valid_A"]}
    valid = (len(status) >= 6 and len(mid) >= 2 and len(out["status"]["cells"]) == 2
             and all(c["valid_A"] for c in out["status"]["cells"].values()))
    r2s = [c["r2_fixed"] for c in out["status"]["cells"].values()]
    if not valid:
        st = "UNCONFIRMED"
    elif all(r >= 0.70 for r in r2s):
        st = "CONFIRMED"
    elif all(r < 0.70 for r in r2s):
        st = "REJECTED"
    else:
        st = "UNCONFIRMED"
    out["status"]["valid"], out["status"]["status"] = valid, st
    # prereg outcome (separate from status)
    bx = [b for b, r in ov["builds"].items() if r["class"] == "bitexact" and not b.startswith("none")]
    bx_ok = all(abs(ov["builds"][b]["overlap"]) < 0.10 and all(
        abs(out["policies"][h]["rows"][b]["share"]) < 0.10 for h in HEAD if b in out["policies"].get(h, {}).get("rows", {}))
        for b in bx)
    judge_ok = all(abs(out["policies"][h]["judge"]["regex_rate"] - out["policies"][h]["judge"]["top1_in_X"]) <= 0.02
                   for h in HEAD if "judge" in out["policies"].get(h, {}))
    out["prereg"] = {"bitexact_builds": bx, "bitexact_ok": bx_ok, "judge_ok": judge_ok,
                     "outcome": "unsettled" if st == "UNCONFIRMED" else
                     ("held" if st == "CONFIRMED" and bx_ok and judge_ok else "failed")}
    json.dump(out, open(HERE / "data" / "summary.json", "w"), indent=1)
    write_matrix(out, ov)
    print(json.dumps({"status": {k: v for k, v in out["status"].items() if k != "builds"}, "prereg": out["prereg"]},
                     indent=1)[:4000])


def write_matrix(out: dict, ov: dict) -> None:
    """proof-matrix.md: one row per cell, each cell's numbers in its own data/cells/<name>.json."""
    cd = HERE / "data" / "cells"
    cd.mkdir(parents=True, exist_ok=True)
    rows = []

    def cell(claim, seed, variant, control, result, name, payload):
        json.dump(payload, open(cd / f"{name}.json", "w"), indent=1)
        rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | data/cells/{name}.json |")

    st = out["status"]
    for h, c in st.get("cells", {}).items():
        sd = h.split("@")[0]
        P = out["policies"][h]
        pay = {"policy": h, "run": P["run"], "status_builds": st["builds"], "cell": c,
               "points": {b: {"overlap": ov["builds"][b]["overlap"], **P["rows"][b]} for b in st["builds"]}}
        cell("headline", sd, h, "none", f"R2 fixed line {c['r2_fixed']:.3f} over {st['n_status']} builds; "
             f"dGAP_A {c['dgap_A']:.3f} (valid {c['valid_A']})", f"headline_{sd}", pay)
        cell("headline", sd, h, "monitor-label permutation (999)", f"p {c['perm_p']:.4f}", f"perm_{sd}",
             {"policy": h, "r2_fixed": c["r2_fixed"], "perm_p": c["perm_p"]})
        for k, v in c["secondary_r2"].items():
            ctl = {"null_shuf": "shuffled-label classifier", "null_rand": "random direction"}.get(k, "none")
            cell("headline" if k.startswith("null") else f"secondary-{k}", sd, h, ctl,
                 f"R2 {v:.3f}" if v is not None else "n/a", f"{k}_{sd}", {"policy": h, "predictor": k, "r2_fixed": v})
        for k in ("r2_without_emu", "r2_with_L7_L21", "r2_all_builds", "r2_nonzero_cloud", "pearson_r2",
                  "r2_high_cluster", "pearson_r_high_cluster", "r2_depth16", "r2_depth16_without_emu", "r2_depth14",
                  "r2_depth14_without_emu"):
            v = c.get(k)
            cell(k.replace("r2_", ""), sd, h, "none", f"{k} {v:.3f}" if v is not None else "n/a", f"{k}_{sd}",
                 {"policy": h, k: v})
        for k in ("pair_sign", "pair_with_seen_sign"):
            v = c[k]
            cell("same-layer-pairs" + ("-with-seen" if "seen" in k else ""), sd, h, "depth held fixed",
                 f"{v['concordant']}/{v['concordant'] + v['discordant']} concordant (ties {v['ties']}), one-sided p "
                 + (f"{v['p_one_sided']:.4f}" if v["p_one_sided"] is not None else "n/a"), f"{k}_{sd}", v)
        for k in ("spearman_le14", "spearman_ge15"):
            v = c[k]
            cell(k, sd, h, "depth bin", f"rho {v['rho']:.3f} (n {v['n']}, p {v['p']:.4f})" if v["rho"] is not None
                 else f"n {v['n']}", f"{k}_{sd}", v)
        if "judge" in P:
            j = P["judge"]
            cell("text-judge", sd, h, "none", f"regex {j['regex_rate']:.3f} vs top-1-in-X {j['top1_in_X']:.3f}",
                 f"judge_{sd}", j)
    for b in out["prereg"]["bitexact_builds"]:
        sh = {h: out["policies"][h]["rows"].get(b, {}).get("share") for h in HEAD if h in out["policies"]}
        cell("bit-exact", "s0,s1", b, "bit-exact build", f"overlap {ov['builds'][b]['overlap']:+.3f}; share "
             + ", ".join(f"{h} {v:+.3f}" for h, v in sh.items() if v is not None), f"bitexact_{tagof(b)}",
             {"build": b, "overlap": ov["builds"][b], "share": sh})
    for name, v in out.get("variants", {}).items():
        ctl = "probe-independent penalty (no switch)" if name.startswith("shuf") else "none"
        r = f"R2 {v['r2_fixed']:.3f} over {v['n']}" if v["r2_fixed"] is not None else f"n {v['n']}"
        cell("variant" if ctl == "none" else "control", name.split("@")[0], name, ctl,
             f"{r}; dGAP_A {v['dgap_A']:.3f} (valid {v['valid_A']})", f"variant_{name.replace('@', '_')}", v)
    hdr = ["# Proof matrix: footprint-overlap", "",
           "Built by analyze.py from data/overlap.json + raw behaviour files. Skeptic rebuild cells are appended below.", "",
           "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    extra = HERE / "skeptics" / "proof-cells.md"
    tail = ["", open(extra).read()] if extra.exists() else []
    open(HERE / "proof-matrix.md", "w").write("\n".join(hdr + rows + tail) + "\n")
    for r in rows:
        assert (HERE / r.rsplit("|", 2)[1].strip()).exists(), r


if __name__ == "__main__":
    main()
