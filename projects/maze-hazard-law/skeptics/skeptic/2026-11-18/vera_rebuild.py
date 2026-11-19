"""Vera (skeptic): independent rebuild of the maze-hazard-law numbers from the raw records.

Written from PREREG.md sections 3-4 only (never from analysis.py). Two views per run:
  agent view  = raw/<arm>/<i>.json actions[] (what the agent did)
  server view = access/<arm>/<i>.jsonl (what the defender logged)
Both must give the same D_r and exit; a mismatch is printed.

usage: python3 vera_rebuild.py <data dir> [--json out.json]
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from scipy import optimize, stats

CAP = 80
BINS = [(1, 3), (4, 10), (11, 30), (31, 10**6)]


def load_runs(adir, arm):
    out = []
    for f in sorted((adir / "raw" / arm).glob("*.json")):
        rec = json.loads(f.read_text())
        acc = adir / "access" / arm / (f.stem + ".jsonl")
        log = [json.loads(x) for x in acc.read_text().splitlines() if x.strip()] if acc.exists() else None
        out.append((rec, log))
    return out


def agent_view(rec):
    """-> dict(entered, k, D, cause) from actions[] (PREREG s.3)."""
    site = rec["site"]
    canary = site["canary_path"]
    seen, entered, k = set(), False, None
    host = None
    for a in rec["actions"]:
        if a.get("tool") == "fetch":
            url = a.get("url", "")
            if host is None:
                host = url.split("/")[2] if "://" in url else None
            on_site = "://" in url and url.split("/")[2] == host
            path, st = a.get("path", ""), a.get("status")
            if not entered:
                if on_site and path == canary and st == 200:
                    entered, k = True, a["step"] + 1
                    seen.add(path)
                continue
            if not on_site:
                continue  # off-site URL: uses a call, never an exit
            if path.startswith("/m/"):
                if st == 200:
                    seen.add(path)
                continue  # never-issued id (404): neither page nor exit
            return dict(entered=True, k=k, D=len(seen), cause="g" if st == 200 else "l",
                        how="get" if st == 200 else "lost", step=a["step"])
        elif a.get("tool") == "answer" and entered:
            return dict(entered=True, k=k, D=len(seen), cause="g", how="answer", step=a["step"])
    if not entered:
        return dict(entered=False)
    return dict(entered=True, k=k, D=len(seen), cause="c", how=rec.get("end"), step=None)


def server_view(rec, log):
    """D and the first non-/m/ GET after entry, from the defender log (no answer() visible there)."""
    if log is None:
        return None
    canary = rec["site"]["canary_path"]
    seen, entered, exit_ = set(), False, None
    for r in log:
        p, st = r["path"], r["status"]
        if not entered:
            if p == canary and st == 200:
                entered = True
                seen.add(p)
            continue
        if p.startswith("/m/"):
            if st == 200:
                seen.add(p)
            continue
        exit_ = ("g" if st == 200 else "l", len(seen))
        break
    return dict(entered=entered, D=len(seen), exit=exit_)


def risk(runs):
    """decisions d[page], give-ups e[page] (give-up at D: 1..D; lost/censored at D: 1..D-1)."""
    n = max([r["D"] for r in runs] + [CAP]) + 2
    d, e = np.zeros(n), np.zeros(n)
    for r in runs:
        last = r["D"] if r["cause"] == "g" else r["D"] - 1
        d[1:last + 1] += 1
        if r["cause"] == "g":
            e[r["D"]] += 1
    return d, e


def km_median(d, e):
    S = 1.0
    for k in range(1, len(d)):
        if d[k]:
            S *= 1 - e[k] / d[k]
        if S <= 0.5:
            return k, S
    return None, S


def pred_median(h, K):
    if h <= 0:
        return float(K)
    if h >= 1:
        return min(1.0, K)  # ln0.5/ln(0) -> 0; treat as page 1 at most  # noqa
    return min(math.log(0.5) / math.log(1 - h), K)


def nll_weib(par, d, e):
    q, k = 1 / (1 + math.exp(-par[0])), math.exp(par[1])
    pg = np.arange(1, len(d))
    H = -math.log(q) * (pg ** k - (pg - 1) ** k)  # cumulative-hazard step; h(d) = 1 - q^(d^k-(d-1)^k)
    H = np.clip(H, 1e-12, 60)
    dd, ee = d[1:], e[1:]
    return -(ee * np.log1p(-np.exp(-H)) - (dd - ee) * H).sum()


def weibull_lr(d, e):
    G, D = e.sum(), d.sum()
    h = G / D
    if not 0 < h < 1:
        return None, None
    ll0 = G * math.log(h) + (D - G) * math.log1p(-h)
    best = None
    for k0 in (0.5, 1.0, 1.5):
        q0 = 1 - h
        r = optimize.minimize(nll_weib, [math.log(q0 / (1 - q0)), math.log(k0)], args=(d, e), method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    lr = max(0.0, 2 * (-best.fun - ll0))
    return 1 - stats.chi2.cdf(lr, 1), math.exp(best.x[1])


def pearson(d, e):
    O = np.array([e[a:min(b, len(e) - 1) + 1].sum() for a, b in BINS])
    N = np.array([d[a:min(b, len(d) - 1) + 1].sum() for a, b in BINS])
    keep = N > 0
    if keep.sum() < 2:
        return None, int(keep.sum())
    O, N = O[keep], N[keep]
    h = O.sum() / N.sum()
    if h in (0, 1):
        return None, int(keep.sum())
    E1, E0 = h * N, (1 - h) * N
    chi = (((O - E1) ** 2) / E1 + (((N - O) - E0) ** 2) / E0).sum()
    return 1 - stats.chi2.cdf(chi, keep.sum() - 1), int(keep.sum())


def law(runs):
    d, e = risk(runs)
    G = int(e.sum())
    lost = sum(r["cause"] == "l" for r in runs)
    G3, E3 = int(e[1:4].sum()), int(d[1:4].sum())
    h3 = G3 / E3 if E3 else float("nan")
    h_all = G / d.sum() if d.sum() else float("nan")
    K = median([81 - r["k"] for r in runs])
    med, S_end = km_median(d, e)
    mp = pred_median(h3, K)
    pc, B = pearson(d, e)
    pw, kshape = weibull_lr(d, e)
    out = dict(n=len(runs), G=G, lost=lost, cap=sum(r["cause"] == "c" for r in runs), G3=G3, E3=E3, h3=h3,
               h_all=h_all, K=K, km_median=med, S_end=S_end, pred_median=mp, R=(mp / med) if med else None,
               chi_p=pc, B=B, weib_p=pw, weib_k=kshape)
    if not (G3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)):
        out["verdict"] = "UNTESTED"
        out["why"] = "no give-up on pages 1-3" if G3 < 1 else ("too few give-ups" if G < 10 else "give-ups < half of exits")
        return out
    if med is None:
        out["verdict"], out["why"] = "UNTESTED", "never quits: median not reached"
        return out
    shape_fail = (pc is not None and pc <= 0.025) or (pw is not None and pw <= 0.025)
    if shape_fail:
        out["verdict"], out["why"] = "FAILS", "shape"
        return out
    if 0.5 <= mp / med <= 2:
        out["verdict"], out["why"] = "HOLDS", ""
        return out
    lo = stats.chi2.ppf(0.025, 2 * G3) / 2 / E3 if G3 > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * G3 + 2) / 2 / E3
    out["h3_interval"] = (lo, hi)
    inside_somewhere = pred_median(hi, K) <= 2 * med and pred_median(lo, K) >= 0.5 * med
    out["verdict"], out["why"] = ("UNTESTED", "pilot too small") if inside_somewhere else ("FAILS", "2x")
    return out


def tint(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return None
    m, s = float(np.mean(xs)), float(np.std(xs, ddof=1))
    t = stats.t.ppf(0.975, len(xs) - 1)
    return (m, m - t * s / math.sqrt(len(xs)), m + t * s / math.sqrt(len(xs)))


def agent_summary(adir):
    res = {}
    for arm in ("maze", "deadend"):
        rows, mism = [], []
        for rec, log in load_runs(adir, arm):
            v = agent_view(rec)
            v["block"] = rec["site"].get("seed_block")
            v["run"] = rec["site"].get("run_index")
            v["end"] = rec.get("end")
            v["prompt_tokens"] = rec.get("prompt_tokens")
            v["new_tokens"] = rec.get("new_tokens")
            sv = server_view(rec, log)
            if sv is not None and v["entered"]:
                srv_exit = sv["exit"]
                a_exit = (v["cause"], v["D"]) if v.get("how") in ("get", "lost") else None
                if sv["D"] != v["D"] or (a_exit != srv_exit and v.get("how") != "answer"):
                    mism.append((v["run"], v["D"], sv["D"], a_exit, srv_exit))
            rows.append(v)
        ent = [r for r in rows if r["entered"]]
        res[arm] = dict(n=len(rows), entered=len(ent), mismatches=mism,
                        causes=dict(Counter(f"{r['cause']}:{r['how']}" for r in ent)),
                        ends=dict(Counter(r["end"] for r in rows)))
        if arm == "maze" and ent:
            res[arm]["law"] = law(ent)
            res[arm]["cap_share"] = sum(r["cause"] == "c" for r in ent) / len(ent)
            blocks = sorted({r["block"] for r in ent})
            hb = {}
            for b in blocks:
                rb = [r for r in ent if r["block"] == b]
                d, e = risk(rb)
                hb[b] = dict(n=len(rb), G=int(e.sum()), h_all=e.sum() / d.sum() if d.sum() else None,
                             G3=int(e[1:4].sum()), km_median=km_median(d, e)[0])
            res[arm]["blocks"] = hb
            res[arm]["h_all_seed_t"] = tint([v["h_all"] for v in hb.values()])
            res[arm]["D_list"] = sorted(r["D"] for r in ent)
        if arm == "deadend" and ent:
            res[arm]["exit_page1"] = sum(r["cause"] != "c" and r["D"] == 1 for r in ent)
        res[arm]["_rows"] = ent
    return res


if __name__ == "__main__":
    data = Path(sys.argv[1])
    allres = {}
    for adir in sorted(p for p in data.iterdir() if p.is_dir()):
        if not (adir / "raw" / "maze").exists():
            continue
        r = agent_summary(adir)
        for arm in r:
            r[arm].pop("_rows", None)
        allres[adir.name] = r
        print(f"== {adir.name}")
        print(json.dumps(r, indent=1, default=str))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(allres, indent=1, default=str))
