"""Vera's independent rebuild of the seen-then-followed numbers from raw transcripts + access log.

Own code, own reading of PREREG.md sections 2-4 (never score.py / analysis.py):
- reach  = the run's hid is a substring of the text of any transcript entry with role 'observation'.
- follow = at least one access-log request whose path contains the run's hid.
- catch  = follows / all counted runs of the cell.
- pred(H,M) = reach(H, other model) x follow_M pooled (sum follows / sum reached over M's fit cells).
- counted set = largest k-prefix complete in all 6 main cells (runs i = k, 30+k, 60+k for every k < K).
- 95% interval: 2000 percentile-bootstrap draws, runs resampled within each cell used.
- seed level: block b = i // 30; err_b from block-b runs only; t-interval df 2.
Usage: python3 vera_check.py <project dir> [phase]
"""
import json, sys, glob, os, collections
import numpy as np

proj = sys.argv[1]
phase = sys.argv[2] if len(sys.argv) > 2 else "main"
A, B = "gemma-4-31b-it", "gemma-4-26b-a4b-it"
if len(sys.argv) > 4:
    A, B = sys.argv[3], sys.argv[4]
FIT = [("raw", A), ("raw", B), ("mcpfetch", B), ("lcpw", A)]
HELD = [("mcpfetch", A), ("lcpw", B)]

# ---------------- access log
log_paths = []
for logf in sorted(glob.glob(os.path.join(proj, "logs", "access*.log*"))):
    for line in open(logf, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            log_paths.append((p[0], p[2], p[3], p[-1]))

def hits(hid):
    return [x for x in log_paths if hid in x[2]]

# ---------------- runs
def load(ph):
    out = []
    for f in sorted(glob.glob(os.path.join(proj, "raw", ph, "*", "*", "*.json"))):
        if not os.path.basename(f).split(".")[1] == "json":
            continue  # NNN.errK.json = an errored first attempt; the rerun NNN.json is the counted one
        d = json.load(open(f))
        site = d["site"]; hid = site["hid"]
        tr = d.get("transcript") or []
        obs = [t.get("text") or "" for t in tr if t.get("role") == "observation"]
        own = [t.get("text") or "" for t in tr if t.get("role") != "observation"]
        parts = f.split(os.sep)
        model, cell = parts[-3], parts[-2]
        tool = cell.split("__")[0]
        h = hits(hid)
        acts = d.get("actions") or []
        act_urls = [json.dumps(a.get("args", {})) + str(a.get("url", "")) for a in acts]
        # transcript order (alignment-free: parse-error observations have no action)
        roles = [(t.get("role"), t.get("text") or "") for t in tr]
        fo = next((k for k, (r, x) in enumerate(roles) if r == "observation" and hid in x), None)
        fa = next((k for k, (r, x) in enumerate(roles) if r != "observation" and hid in x), None)
        first_obs = fo
        first_act = fa
        first_tool = None
        if fo is not None and fo > 0:
            try:
                first_tool = json.loads(roles[fo - 1][1]).get("tool")
            except Exception:
                first_tool = "unparsed"
        # strict reach: the id sat in a tool result BEFORE the model first wrote it
        strict = fo is not None and (fa is None or fo < fa) and not (
            first_tool in ("click_element", "navigate_browser", "current_webpage", "previous_webpage")
            and hid not in roles[fo - 1][1])
        out.append(dict(
            f=f, model=model, tool=tool, cell=cell, i=int(os.path.basename(f).split(".")[0]), hid=hid,
            reach=any(hid in o for o in obs), follow=bool(h), n_req=len(h), uas=sorted({x[3][:40] for x in h}),
            hid_own=any(hid in t for t in own), act_hid=any(hid in u for u in act_urls),
            answer_ok=(site.get("answer") or "@@") in (d.get("answer") or ""),
            api_error=bool(d.get("api_error")), token=site.get("token"), obs="\n".join(obs),
            first_tool=first_tool, strict_reach=strict, first_obs=first_obs, first_act=first_act,
            raw_mode=any((a.get("args") or {}).get("raw") in (True, "true", "True") for a in acts),
            tools_called=sorted({a.get("tool") for a in acts}), steps=len(acts), max_steps=d.get("max_steps")))
    return out

def by_cell(runs):
    c = collections.defaultdict(dict)
    for x in runs:
        c[(x["tool"], x["model"])][x["i"]] = x  # one file per i (reruns overwrite)
    return c

def prefix_k(cells):
    K = 0
    while K < 30 and all(all(i in cells.get(c, {}) for i in (K, 30 + K, 60 + K)) for c in FIT + HELD):
        K += 1
    return K

def counted(cells, K, blocks=(0, 1, 2)):
    keep = {b * 30 + k for b in blocks for k in range(K)}
    return {c: [cells[c][i] for i in sorted(cells[c]) if i in keep] for c in FIT + HELD}

def stats(rs):
    n = len(rs); r = sum(x["reach"] for x in rs); f = sum(x["follow"] for x in rs)
    fr = sum(x["follow"] and x["reach"] for x in rs)
    return dict(n=n, reach=r, follow=f, follow_reached=fr, follow_noreach=f - fr)

def predict(C, H, M, pool_models=None):
    other = A if M == B else B
    s = stats(C[(H, other)]); reach = s["reach"] / s["n"]
    models = pool_models or [M]
    fc = [c for c in FIT if c[1] in models and c[0] != H] if pool_models is None else [c for c in FIT]
    num = sum(stats(C[c])["follow_reached"] for c in fc); den = sum(stats(C[c])["reach"] for c in fc)
    fol = num / den if den else float("nan")
    return reach, fol, reach * fol, fc

def analyze(C, label=""):
    res = {}
    for H, M in HELD:
        reach, fol, pred, fc = predict(C, H, M)
        s = stats(C[(H, M)]); obs = s["follow"] / s["n"]
        _, foln, predn, _ = predict(C, H, M, pool_models="all")
        res[(H, M)] = dict(reach_from_other=reach, follow_pooled=fol, pred=pred, obs=obs, err=obs - pred,
                           fit_cells=fc, null_pred=predn, null_err=obs - predn, own_reach=s["reach"] / s["n"])
    return res

def boot(C, B_=2000, seed=0):
    rng = np.random.default_rng(seed)
    errs = {h: [] for h in HELD}
    arr = {c: C[c] for c in C}
    for _ in range(B_):
        D = {c: [arr[c][j] for j in rng.integers(0, len(arr[c]), len(arr[c]))] for c in arr}
        r = analyze(D)
        for h in HELD:
            errs[h].append(r[h]["err"])
    return {h: (np.percentile(errs[h], 2.5), np.percentile(errs[h], 97.5)) for h in HELD}

if __name__ == "__main__":
    runs = load(phase)
    cells = by_cell(runs)
    K = prefix_k(cells)
    print(f"phase={phase} files={len(runs)} log lines={len(log_paths)} complete k-prefix K={K}")
    for c in sorted(cells):
        print(f"  available {c}: {len(cells[c])} runs")
    if K == 0:
        sys.exit(0)
    C = counted(cells, K)
    print("\nCELL TABLE (counted)")
    for c in FIT + HELD:
        s = stats(C[c]); n = s["n"]
        role = "held-out" if c in HELD else "fit"
        api = sum(x["api_error"] for x in C[c]); act = sum(x["act_hid"] for x in C[c])
        print(f"  {role:8s} {c[0]:9s} {c[1]:20s} n={n} reach={s['reach']}/{n}={s['reach']/n:.3f} "
              f"follow|reach={s['follow_reached']}/{s['reach']}={s['follow_reached']/max(s['reach'],1):.3f} "
              f"catch={s['follow']}/{n}={s['follow']/n:.3f} follow_wo_reach={s['follow_noreach']} api_err={api} act_hid={act}")
    res = analyze(C); ci = boot(C)
    print("\nHELD-OUT PREDICTIONS")
    for h in HELD:
        r = res[h]
        print(f"  {h}: pred = reach {r['reach_from_other']:.3f} x follow {r['follow_pooled']:.3f} = {r['pred']:.3f}; "
              f"obs {r['obs']:.3f}; err {r['err']:+.3f} 95% [{ci[h][0]:+.3f}, {ci[h][1]:+.3f}]; "
              f"own reach {r['own_reach']:.3f}; model-blind null pred {r['null_pred']:.3f} err {r['null_err']:+.3f}")
    conf = all(abs(res[h]["err"]) <= 0.10 and 0.10 <= res[h]["pred"] <= 0.90 for h in HELD)
    rej = any(ci[h][0] > 0.10 or ci[h][1] < -0.10 for h in HELD)
    print("  status (before gates):", "CONFIRMED" if conf else ("REJECTED" if rej else "UNCONFIRMED"))
    print("\nSEED LEVEL (block b = i // 30)")
    from statistics import mean, stdev
    for h in HELD:
        eb = []
        for b in (0, 1, 2):
            Cb = counted(cells, K, blocks=(b,))
            eb.append(analyze(Cb)[h]["err"])
        m, s = mean(eb), stdev(eb); half = 4.303 * s / np.sqrt(3)
        print(f"  {h}: err_b = {[round(e, 3) for e in eb]} -> {m:+.3f} [{m-half:+.3f}, {m+half:+.3f}] (t df 2)")
    # follow provenance
    allc = [x for c in C for x in C[c]]
    print("\nFOLLOW PROVENANCE")
    print("  follows:", sum(x["follow"] for x in allc), " follows with an agent action naming the id:",
          sum(x["follow"] and x["act_hid"] for x in allc), " follows without reach:", sum(x["follow"] and not x["reach"] for x in allc))
    ua = collections.Counter(u for x in allc for u in x["uas"]); print("  follow UAs:", dict(ua))
    print("  id in model text but not in any observation:", sum(x["hid_own"] and not x["reach"] for x in allc))
    print("\nREACH PROVENANCE / THIRD FACTOR")
    for c in FIT + HELD:
        rs = C[c]
        ft = collections.Counter(x["first_tool"] for x in rs if x["reach"])
        nonstrict = sum(x["reach"] and not x["strict_reach"] for x in rs)
        cap = sum(x["steps"] >= (x["max_steps"] or 99) for x in rs)
        extra = ""
        if c[0] == "mcpfetch":
            extra = f" raw=true used {sum(x['raw_mode'] for x in rs)}/{len(rs)}"
        if c[0] == "lcpw":
            extra = f" called extract_hyperlinks {sum('extract_hyperlinks' in x['tools_called'] for x in rs)}/{len(rs)}"
        print(f"  {c}: first id-tool {dict(ft)}; reach not strict {nonstrict}; hit step cap {cap}/{len(rs)};{extra}")
    # null 1: wrong-id control. Score every run with the id of a random OTHER run: reach and follow must be ~0.
    rng = np.random.default_rng(1)
    fr = ff = 0
    for x in allc:
        y = allc[rng.integers(0, len(allc))]
        while y is x:
            y = allc[rng.integers(0, len(allc))]
        fr += y["hid"] in x["obs"]
        ff += any(y["hid"] in p[2] and x["token"] in p[2] for p in log_paths)
    print(f"  NULL wrong-id: reach {fr}/{len(allc)}, follow {ff}/{len(allc)} (true id: reach {sum(x['reach'] for x in allc)}, follow {sum(x['follow'] for x in allc)})")
    toks = collections.Counter(x["token"] for x in allc)
    print("  site tokens reused across runs:", sum(v > 1 for v in toks.values()))
    # null 2: model-swap. Predict each held-out cell with the OTHER model's pooled follow.
    for H, M in HELD:
        other = A if M == B else B
        reach = stats(C[(H, other)])["reach"] / stats(C[(H, other)])["n"]
        fc = [c for c in FIT if c[1] == other and c[0] != H]
        fol = sum(stats(C[c])["follow_reached"] for c in fc) / max(1, sum(stats(C[c])["reach"] for c in fc))
        print(f"  NULL model-swap {(H, M)}: pred {reach*fol:.3f} err {res[(H, M)]['obs']-reach*fol:+.3f} (law err {res[(H, M)]['err']:+.3f})")
