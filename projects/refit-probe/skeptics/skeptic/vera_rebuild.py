"""Vera's independent rebuild of refit-probe (PREREG.md s6-8) from the raw files only.

Reads data/answers_<p>.jsonl, data/labels_<p>.jsonl, data/acts_<p>.npz, data/probes.json. Does NOT read build's
refit.py or results/. Own draw seed (7, not 1019), own AUROC (vera_core.auroc), own bootstrap.

    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_rebuild.py [--quick]
-> skeptic/vera_rebuild.json, printed table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, balanced_draw, catch_at_fpr, fit_lr, score  # noqa: E402

_H = Path(__file__).resolve()
# works from swarm-<room>/skeptic/ and from kran-research/projects/refit-probe/skeptics/skeptic/
D = next(d for d in (_H.parents[2] / "data", _H.parents[2] / "kran-research/projects/refit-probe/data") if d.exists())
OUT = Path(__file__).resolve().parent
POL = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1", "lam0_s0", "lam0_s1"]
LAM1 = ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
NS = [25, 50, 100, 200, 400]
QUICK = "--quick" in sys.argv
SEED = 7


def jl(p):
    return [json.loads(l) for l in open(p)]


def load(p):
    if not (D / f"acts_{p}.npz").exists() or not (D / f"labels_{p}.jsonl").exists():
        return None
    A, L = jl(D / f"answers_{p}.jsonl"), jl(D / f"labels_{p}.jsonl")
    Z = np.load(D / f"acts_{p}.npz")
    n = len(A)
    if len(L) != n:
        print(f"SKIP {p}: {len(L)} labels for {n} answers (labelling not finished)")
        return None
    assert n == len(Z["own_L12"]), (p, len(A), len(Z["own_L12"]))
    for i, (a, l) in enumerate(zip(A, L)):
        assert a["idx"] == l["idx"] == i and a["prompt_id"] == l["prompt_id"] == str(Z["prompt_id"][i]), (p, i)
    y = np.array([-1 if l["y_judge"] is None else l["y_judge"] for l in L])
    y2 = np.array([-1 if l.get("y_judge2") is None else l["y_judge2"] for l in L])
    ys = np.array([l["y_string"] for l in L])
    return dict(
        pid=np.array([a["prompt_id"] for a in A]), split=np.array([a["split"] for a in A]),
        text=[a["response"] for a in A], y=y, y2=y2, ys=ys,
        own=Z["own_L12"].astype(np.float64), basereads=Z["base_L12"].astype(np.float64),
        pA=np.array([a["pA"] for a in A]), flagA=np.array([a["flagA"] for a in A]),
        pEns=np.array([a["pEns"] for a in A]), flagEns=np.array([a["flagEns"] for a in A]),
        pB2=np.array([a["pB2"] for a in A]),
    )


def main():
    probes = json.load(open(D / "probes.json"))
    PA = probes["A"]
    data = {p: load(p) for p in POL}
    data = {p: d for p, d in data.items() if d is not None}
    print("policies with labels+acts:", list(data))
    res = {"seed": SEED, "policies": {}}

    # ---- integrity: test prompts identical across policies, disjoint from pool, no duplicates
    test_ids = None
    for p, d in data.items():
        te = d["pid"][d["split"] == "test"]
        assert (d["split"][:400] == "test").all() and (d["split"][400:] == "pool").all(), p
        assert len(set(d["pid"])) == len(d["pid"]), f"{p}: duplicate prompt ids"
        assert not set(te) & set(d["pid"][d["split"] == "pool"]), f"{p}: test/pool overlap"
        if test_ids is None:
            test_ids = te
        assert (te == test_ids).all(), f"{p}: test order differs"
    pools = {p: set(d["pid"][d["split"] == "pool"]) for p, d in data.items()}
    print("integrity OK: 400 shared test prompts, pool disjoint, no duplicate prompt ids")

    rng0 = np.random.default_rng(SEED)
    for p, d in data.items():
        te, po = d["split"] == "test", d["split"] == "pool"
        yt, ypool = d["y"][te], d["y"][po]
        r = res["policies"][p] = {}
        # frozen probe A rebuilt from probes.json on own acts (float16 acts -> small error expected)
        x = (d["own"] - np.array(PA["mu"])) / np.array(PA["sd"])
        pA_me = 1 / (1 + np.exp(-(x @ np.array(PA["w"]) + PA["b"])))
        r["pA_rebuild_maxabs"] = float(np.abs(pA_me - d["pA"]).max())
        r["flagA_mismatch"] = int(((pA_me > PA["thr"]).astype(int) != d["flagA"]).sum())
        r["counts_test"] = {"dec": int((yt == 1).sum()), "hon": int((yt == 0).sum()), "other": int((yt < 0).sum())}
        r["counts_pool"] = {"dec": int((ypool == 1).sum()), "hon": int((ypool == 0).sum())}
        both = (yt >= 0) & (d["y2"][te] >= 0)
        r["judge_vs_haiku_agree_test"] = float((yt[both] == d["y2"][te][both]).mean())
        sv = yt >= 0
        r["judge_vs_string_agree_test"] = float((yt[sv] == d["ys"][te][sv]).mean())
        old = d["pEns"] if p.startswith("ens") else d["pA"]
        r["old_probe_auroc_test"] = auroc(yt, old[te])
        r["pA_auroc_test"] = auroc(yt, d["pA"][te])
        r["pB2_truefalse_auroc_test"] = auroc(yt, d["pB2"][te])
        oldflag = d["flagEns"] if p.startswith("ens") else d["flagA"]
        r["old_probe_catch_at_old_thr"] = float(oldflag[te][yt == 1].mean())
        r["old_probe_catch5"] = catch_at_fpr(yt[yt >= 0], old[te][yt >= 0])

        # BoW vocabulary from the pool text only
        Xo_pool, Xo_te = d["own"][po], d["own"][te]
        Xb_pool, Xb_te = d["basereads"][po], d["basereads"][te]
        txt_pool = [t for t, k in zip(d["text"], po) if k]
        txt_te = [t for t, k in zip(d["text"], te) if k]
        cv = CountVectorizer(min_df=2, binary=True).fit(txt_pool)
        W_pool, W_te = cv.transform(txt_pool), cv.transform(txt_te)
        len_pool = np.log1p(np.array([len(t.split()) for t in txt_pool]))[:, None]
        len_te = np.log1p(np.array([len(t.split()) for t in txt_te]))[:, None]

        r["arms"] = {}
        r["scores200"] = {"own": [], "held": []}
        for n in NS:
            ndraw = (6 if QUICK else (50 if n <= 50 else 20))
            arm = {k: [] for k in ["own", "held", "bow", "len", "shuf", "own_catch5", "held_catch5"]}
            ok = True
            for _ in range(ndraw):
                idx = balanced_draw(ypool, n, rng0)
                if idx is None:
                    ok = False
                    break
                yf = ypool[idx]
                m_own = fit_lr(Xo_pool[idx], yf)
                s_own = score(m_own, Xo_te)
                m_h = fit_lr(Xb_pool[idx], yf)
                s_h = score(m_h, Xb_te)
                arm["own"].append(auroc(yt, s_own))
                arm["held"].append(auroc(yt, s_h))
                arm["own_catch5"].append(catch_at_fpr(yt[yt >= 0], s_own[yt >= 0]))
                arm["held_catch5"].append(catch_at_fpr(yt[yt >= 0], s_h[yt >= 0]))
                bw = LogisticRegression(C=0.05, max_iter=2000).fit(W_pool[idx], yf)
                arm["bow"].append(auroc(yt, bw.decision_function(W_te)))
                ln = LogisticRegression(max_iter=2000).fit(len_pool[idx], yf)
                arm["len"].append(auroc(yt, ln.decision_function(len_te)))
                if n == 200:
                    arm["shuf"].append(auroc(yt, score(fit_lr(Xo_pool[idx], rng0.permutation(yf)), Xo_te)))
                    r["scores200"]["own"].append(s_own)
                    r["scores200"]["held"].append(s_h)
            if not ok:
                r["arms"][n] = None
                continue
            r["arms"][n] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in arm.items() if v}
            r["arms"][n]["ndraw"] = ndraw
        # random-direction null: |AUROC - 0.5| of 1000 random unit directions on own acts
        R = rng0.standard_normal((1000, d["own"].shape[1]))
        ra = np.array([auroc(yt, Xo_te @ v) for v in R])
        r["random_dir_auroc_p95_p99"] = [float(np.percentile(np.maximum(ra, 1 - ra), q)) for q in (95, 99)]
        print(p, json.dumps({k: v for k, v in r.items() if k not in ("arms", "scores200")}))
        for n in NS:
            a = r["arms"][n]
            print("   n=%3d" % n, "missing" if a is None else " ".join(
                f"{k}={a[k][0]:.3f}" for k in ["own", "held", "bow", "len", "shuf", "own_catch5"] if k in a))

    # ---- verdict: gap = base - policy refit AUROC at n=200 (mean over draws), paired bootstrap over test prompts
    if "base" in data:
        yb = data["base"]["y"][:400]
        Sb = np.array(res["policies"]["base"]["scores200"]["own"])
        rngb = np.random.default_rng(1019)
        B = 200 if QUICK else 1000
        boots = rngb.integers(0, 400, (B, 400))
        res["verdict"] = {}
        for p in [q for q in LAM1 if q in data]:
            yp = data[p]["y"][:400]
            for arm in ["own", "held"]:
                Sp = np.array(res["policies"][p]["scores200"][arm])
                Sbb = Sb if arm == "own" else np.array(res["policies"]["base"]["scores200"]["held"])
                pt = np.mean([auroc(yb, s) for s in Sbb]) - np.mean([auroc(yp, s) for s in Sp])
                g = np.empty(B)
                for i, ix in enumerate(boots):
                    g[i] = np.mean([auroc(yb[ix], s[ix]) for s in Sbb]) - np.mean([auroc(yp[ix], s[ix]) for s in Sp])
                lo, hi = np.percentile(g, [2.5, 97.5])
                res["verdict"][f"{p}:{arm}"] = {"gap": float(pt), "lo": float(lo), "hi": float(hi)}
                print(f"gap base-{p} [{arm}] = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    for p in res["policies"].values():
        p.pop("scores200", None)
    (OUT / ("vera_rebuild_quick.json" if QUICK else "vera_rebuild.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
