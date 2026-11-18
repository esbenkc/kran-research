"""Vera (skeptic): independent rebuild of the maze-cost frozen rule from RAW files only.

Inputs (Bruno's raw contract, #24762): <data>/raw/<arm>/<i>.json  (per-call token counts + action)
                                       <data>/access/<arm>/<i>.jsonl (every server request)
I do NOT use harness flags (kind, depth, end reason); depth, entry, leave and spend are rebuilt from
request paths + call actions. Frozen rule: PREREG.md l.119-171 (pushed 201fcc0e1).

usage: python3 vera_check.py <data dir> [--prices in,out] [--json out.json]
"""
import argparse, glob, json, math, os, re, sys
from collections import Counter
import numpy as np

CAP = 12
MAZE_RE = re.compile(r"^/m/([^/?#]+)\.html")


def load_runs(data, arm):
    runs = []
    for f in sorted(glob.glob(os.path.join(data, "raw", arm, "*.json")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        i = os.path.splitext(os.path.basename(f))[0]
        raw = json.load(open(f))
        af = os.path.join(data, "access", arm, f"{i}.jsonl")
        acc = [json.loads(l) for l in open(af) if l.strip()] if os.path.exists(af) else []
        acc.sort(key=lambda r: r.get("t", 0))  # handler logs after sending; order by request time
        runs.append((i, raw, acc))
    return runs


def action_of(call):
    a = call.get("action")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return ("bad", None)
    if not isinstance(a, dict):
        return ("bad", None)
    tool = a.get("tool") or a.get("name") or a.get("type") or a.get("action")
    if tool in ("fetch", "get", "open"):
        return ("fetch", a.get("url") or (a.get("args") or {}).get("url"))
    if tool in ("answer", "final", "submit"):
        return ("answer", a.get("text") or a.get("answer") or (a.get("args") or {}).get("text"))
    return ("bad", None)


def path_of(url):
    if url is None:
        return None
    u = re.sub(r"^https?://[^/]+", "", url.strip())
    return u if u.startswith("/") else "/" + u


def _m(call):
    return call.get("meta") or call


def tokens(call):
    m = _m(call)
    return (m.get("prompt_tokens") or 0) + (m.get("candidates_tokens") or 0) + (m.get("thoughts_tokens") or 0)


def rebuild_run(raw, acc, notes):
    """Return dict with entered, k (1-based entry call), D, left, leave_depth, S, T{j}, M, defender (cpu_ns, bytes, n_req)."""
    calls = sorted(raw.get("calls") or raw["steps"], key=lambda c: c["step"])
    # align fetch actions <-> access log requests (in order). Requests = what the server saw.
    reqs = [r for r in acc]
    fetch_calls = [ci for ci, c in enumerate(calls) if action_of(c)[0] == "fetch"]
    if len(fetch_calls) != len(reqs):
        notes["fetch_req_mismatch"] += 1
    req_of_call = {}
    for n, ci in enumerate(fetch_calls):
        if n < len(reqs):
            req_of_call[ci] = reqs[n]
    seen_ids, k, depth_before = [], None, []
    out = dict(entered=False, k=None, D=0, left=False, leave_depth=None, S=0, T=Counter(), M=None,
               cpu_ns=0, bytes=0, n_req=0, n_calls=len(calls), end=None, tok_in=0, tok_out=0)
    for ci, c in enumerate(calls):
        depth_before.append(len(seen_ids))
        kind, arg = action_of(c)
        in_episode = k is not None
        if in_episode:
            j = depth_before[ci]
            t = tokens(c)
            out["S"] += t
            out["T"][j] += t
            out["tok_in"] += _m(c).get("prompt_tokens") or 0
            out["tok_out"] += (_m(c).get("candidates_tokens") or 0) + (_m(c).get("thoughts_tokens") or 0)
        r = req_of_call.get(ci)
        rpath = r.get("path") if r else path_of(arg) if kind == "fetch" else None
        m = MAZE_RE.match(rpath or "") if kind == "fetch" else None
        ok = r is not None and int(r.get("status", 200)) == 200
        if in_episode and r is not None and m:  # PREREG s3: defender cost = maze requests only
            out["cpu_ns"] += r.get("cpu_ns") or 0
            out["bytes"] += r.get("bytes") or 0
            out["n_req"] += 1
        if kind == "fetch" and m:
            if k is None:
                if ok:
                    k = ci + 1  # entry call index, 1-based
                    out["entered"], out["k"] = True, k
                    out["cpu_ns"] += r.get("cpu_ns") or 0
                    out["bytes"] += r.get("bytes") or 0
                    out["n_req"] += 1
                    seen_ids.append(m.group(1))
            elif ok and m.group(1) not in seen_ids:
                seen_ids.append(m.group(1))
            continue
        if in_episode and (kind == "answer" or (kind == "fetch" and not m)):
            out["left"] = True
            out["leave_depth"] = depth_before[ci]
            out["end"] = "answer" if kind == "answer" else "get_out"
            break
    out["D"] = len(seen_ids)
    if out["entered"]:
        out["M"] = CAP - out["k"]
        if not out["left"]:
            out["end"] = "cap" if len(calls) >= CAP else "censored_other"
    else:
        out["end"] = "no_entry"
    return out


def km_median(S, cens):
    """Smallest S at which KM survival <= 0.5. cens True = censored."""
    order = np.lexsort((np.asarray(cens), np.asarray(S)))  # events before censorings at ties
    S = np.asarray(S, float)[order]; cens = np.asarray(cens)[order]
    n, surv = len(S), 1.0
    at_risk = n
    i = 0
    while i < n:
        s = S[i]; d = 0; c = 0
        while i < n and S[i] == s:
            if cens[i]: c += 1
            else: d += 1
            i += 1
        if d:
            surv *= (1 - d / at_risk)
            if surv <= 0.5:
                return s
        at_risk -= d + c
    return None


def law(runs):
    ent = [r for r in runs if r["entered"]]
    L3 = sum(1 for r in ent if r["left"] and 1 <= r["leave_depth"] <= 3)
    E3 = 0
    for r in ent:
        if r["left"]:
            E3 += min(r["leave_depth"], 3)
        else:
            E3 += min(max(r["D"] - 1, 0), 3)
    h = L3 / E3 if E3 else float("nan")
    dstar = None
    if h and h > 0:
        dstar = 1 if h >= 1 else math.ceil(math.log(0.5) / math.log(1 - h))
        # exact boundary: smallest d with (1-h)^d <= 0.5
        while dstar > 1 and (1 - h) ** (dstar - 1) <= 0.5:
            dstar -= 1
    xs, ys = [], []
    for r in ent:
        for j, t in r["T"].items():
            if 1 <= j <= 3:
                xs.append(j); ys.append(t)
    a = b = None
    if len(set(xs)) >= 2:
        b, a = np.polyfit(xs, ys, 1)
    P = sum(a + b * j for j in range(1, dstar + 1)) if (dstar and a is not None) else None
    O = km_median([r["S"] for r in ent], [not r["left"] for r in ent]) if ent else None
    K = float(np.median([r["M"] for r in ent])) if ent else None
    surv_cap = (1 - h) ** K if (ent and h == h) else None
    if not ent or dstar is None:
        outcome = "never quits: spend set by the cap"
    elif dstar <= 3:
        outcome = "quits inside the pilot"
    elif surv_cap >= 0.5 or O is None:
        outcome = "never quits: spend set by the cap"
    else:
        outcome = "TESTED"
    R = P / O if (P is not None and O) else None
    return dict(n_entered=len(ent), L3=L3, E3=E3, h=h, d_star=dstar, a=a, b=b, P=P, O=O, K=K,
                surv_cap=surv_cap, outcome=outcome, R=R,
                in_band=(R is not None and 0.5 <= R <= 2))


def lr_constancy(runs):
    from scipy.stats import chi2
    L = [0, 0]; E = [0, 0]
    for r in runs:
        if not r["entered"]:
            continue
        ndec = r["leave_depth"] if r["left"] else max(r["D"] - 1, 0)
        for d in range(1, ndec + 1):
            g = 0 if d <= 3 else 1
            E[g] += 1
            if r["left"] and d == r["leave_depth"]:
                L[g] += 1
    def ll(l, e):
        if e == 0: return 0.0
        p = l / e
        return (l * math.log(p) if l else 0) + ((e - l) * math.log(1 - p) if e - l else 0)
    if E[1] == 0:
        return dict(L=L, E=E, p=None)
    stat = 2 * (ll(L[0], E[0]) + ll(L[1], E[1]) - ll(sum(L), sum(E)))
    return dict(L=L, E=E, h13=L[0] / E[0] if E[0] else None, h4p=L[1] / E[1], stat=stat, p=float(chi2.sf(stat, 1)))


def usd(r, pin, pout):
    agent = r["tok_in"] * pin / 1e6 + r["tok_out"] * pout / 1e6
    cpu_ms = r["cpu_ns"] / 1e6
    defender = cpu_ms * 0.02 / 1e6 + r["bytes"] * 0.085 / 1e9 + r["n_req"] * 0.30 / 1e6
    return agent, defender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--prices", default="0.20,0.40")
    ap.add_argument("--json")
    ap.add_argument("--only")
    a = ap.parse_args()
    pin, pout = map(float, a.prices.split(","))
    res = {}
    arms_found = sorted(os.listdir(os.path.join(a.data, "raw"))) if os.path.isdir(os.path.join(a.data, "raw")) else []
    for arm in arms_found:
        if a.only and arm != a.only:
            continue
        if not os.path.isdir(os.path.join(a.data, "raw", arm)):
            continue
        notes = Counter()
        runs = []
        for i, raw, acc in load_runs(a.data, arm):
            r = rebuild_run(raw, acc, notes)
            r["i"] = int(i); r["block"] = raw.get("seed_block", (raw.get("site") or {}).get("seed_block"))
            st = raw.get("calls") or raw.get("steps") or []
            r["first_action"] = (st[0].get("raw") or st[0].get("text") or json.dumps(st[0].get("action"), sort_keys=True)) if st else None
            r["correct"] = raw.get("correct")
            runs.append(r)
        ends = Counter(r["end"] for r in runs)
        row = dict(n=len(runs), ends=dict(ends), notes=dict(notes))
        if arm in ("gate", "visible"):
            row["correct"] = sum(bool(r["correct"]) for r in runs)
            res[arm] = row; continue
        row["law"] = law(runs)
        row["lr"] = lr_constancy(runs)
        from scipy.stats import beta
        Lt, Et = sum(row["lr"]["L"]), sum(row["lr"]["E"])
        up = float(beta.ppf(0.95, Lt + 1, Et - Lt)) if Et > Lt else 1.0  # one-sided CP 95% (D5)
        lo = float(beta.ppf(0.025, Lt, Et - Lt + 1)) if Lt > 0 else 0.0
        row["leave_rate_all"] = dict(L=Lt, E=Et, rate=Lt / Et if Et else None, cp95=[lo, up],
                                     median_depth_lower=(math.ceil(math.log(0.5) / math.log(1 - up) - 1e-12)) if 0 < up < 1 else None)
        ent_ = [r for r in runs if r["entered"]]
        nl = sum(r["left"] for r in ent_)
        row["run_level_leave"] = dict(left=nl, n=len(ent_),
            up95=float(beta.ppf(0.95, nl + 1, len(ent_) - nl)) if len(ent_) > nl else 1.0)
        ent = [r for r in runs if r["entered"]]
        rat = [usd(r, pin, pout) for r in ent]
        ratios = [ag / de for ag, de in rat if de > 0]
        row["ratio_median"] = float(np.median(ratios)) if ratios else None
        row["ratio_band"] = [row["ratio_median"] * 0.1, row["ratio_median"] * 10] if ratios else None
        row["leave_share"] = sum(r["left"] for r in ent) / len(ent) if ent else None
        row["share_depth4"] = sum(r["D"] >= 4 for r in ent) / len(ent) if ent else None
        row["median_S"] = float(np.median([r["S"] for r in ent])) if ent else None
        row["D_hist"] = dict(Counter(r["D"] for r in ent))
        row["blocks"] = {}
        for b in sorted({r["block"] for r in runs}, key=lambda x: (x is None, x)):
            br = [r for r in runs if r["block"] == b]
            row["blocks"][str(b)] = dict(n=len(br), n_entered=sum(r["entered"] for r in br), **{k: law(br)[k] for k in ("h", "d_star", "P", "O", "R", "outcome")})
        row["first_actions_distinct"] = len({r["first_action"] for r in runs})
        row["runs"] = [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()} for r in runs]
        res[arm] = row
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
    for arm, row in res.items():
        print(f"== {arm}: n={row['n']} ends={row['ends']} notes={row['notes']} correct={row.get('correct')}")
        if "law" in row:
            L = row["law"]
            print("   law:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in L.items()})
            print("   lr:", row["lr"])
            print("   leave_rate_all:", row["leave_rate_all"], "run_level:", row["run_level_leave"])
            print("   ratio_median(usd)=", row["ratio_median"], "band", row["ratio_band"], "leave_share", row["leave_share"],
                  "share_depth4", row["share_depth4"], "median_S", row["median_S"])
            print("   D_hist", dict(sorted(row["D_hist"].items())), "first_actions_distinct", row["first_actions_distinct"])
            for b, v in row["blocks"].items():
                print("   block", b, v)


if __name__ == "__main__":
    main()
