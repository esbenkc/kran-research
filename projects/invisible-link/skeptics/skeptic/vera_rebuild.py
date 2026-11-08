"""Vera: independent rebuild of the invisible-link headline numbers from raw files only.
No build code imported. Inputs: raw/manifest.jsonl, raw/<phase>/... run files (transcript), logs/access.log.
Run: python3 vera_rebuild.py <project_dir> <phase> [B]
Definitions (PREREG 816beab8 s3-s5):
  follow  = a GET with status 200 of the run's own hidden_path in access.log (own parser).
  exposed = the run's hid string occurs in at least one observation the model received.
  title gap = mean over agents of [fit rate - nonfit rate], STUCK, hidden arms, raw+md pooled, exposed runs.
  CI = 2.5/97.5 percentiles of a cluster bootstrap: within each agent x template stratum resample the paraphrase
       clusters with replacement; a cluster brings all its runs (both arms, both views).
  YES if gap >= .05 and lo > 0; elif hi < .15 NO; else NOT SETTLED.
"""
import json, os, re, sys, collections
import numpy as np

proj, phase = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
rng = np.random.default_rng(20261108)

man_all = [json.loads(l) for l in open(os.path.join(proj, "raw/manifest.jsonl"))]
by_token = {m["token"]: m for m in man_all if m.get("phase") == phase}
# source of truth = the run files on disk; the manifest only adds paraphrase / i (looked up by the file's own token)
import glob as _glob
files = sorted(_glob.glob(os.path.join(proj, "raw", phase, "**", "*.json"), recursive=True))
man = []
for fp in files:
    s = json.load(open(fp))["site"]
    m = by_token.get(s["token"])
    if m is None:
        print("run file with no manifest row:", fp); continue
    m = dict(m, raw_file=os.path.relpath(fp, proj))
    man.append(m)
stale = len(by_token) - len(man)
print(f"manifest rows for phase {phase}: {len(by_token)}; run files: {len(files)}; manifest rows with no live run file (stale/overwritten): {stale}")

gets = collections.defaultdict(list)
by_tok = collections.defaultdict(list)  # token -> ordered [(method, path, status)] from the log
for line in open(os.path.join(proj, "logs/access.log")):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 5:
        p = f[3].split("?")[0]
        gets[p].append((f[2], f[4], f[0]))
        parts = p.split("/")
        if len(parts) > 2:
            by_tok[parts[1]].append((f[2], p, f[4], f[5] if len(f) > 5 else ""))
own = {m["hidden_path"]: m for m in man}

runs = []
dup = collections.Counter()
for m in man:
    d = json.load(open(os.path.join(proj, m["raw_file"])))
    obs = [t["text"] for t in d["transcript"] if t["role"] != "assistant"]
    exposed = any(m["hid"] in o for o in obs)
    follow = any(meth == "GET" and st == "200" for meth, st, _ in gets.get(m["hidden_path"], []))
    head_only = any(meth != "GET" for meth, st, _ in gets.get(m["hidden_path"], [])) and not follow
    # trace: fetched paths with token and hid masked
    acts = [a for a in d.get("actions", [])]
    trace = tuple((a.get("tool"), (a.get("path") or "").replace(m["token"], "T").replace(m["hid"], "H"))
                  for a in acts)
    level = m["cell"].split("-")[0]  # stuck | free | rstuck | rfree | ...
    view, arm = m["view"], m["arm"]
    # follow index from the log: position of the first GET of the hidden path among this run's GETs
    seq = [p for meth, p, st, ua in by_tok.get(m["token"], []) if meth == "GET"]
    fidx = seq.index(m["hidden_path"]) + 1 if m["hidden_path"] in seq else None
    route_pages = {"plans.html", "developers.html", "admissions.html", "tickets.html", "rates.html", "rooms.html"}
    route_before = (fidx is not None and any(p.split("/")[-1] in route_pages for p in seq[:fidx - 1]))
    runs.append(dict(model=m["model"], cell=m["cell"], level=level, stuck=(level == "stuck"), view=view, arm=arm,
                     template=m["template"], para=m["paraphrase"], exposed=exposed, follow=follow,
                     head_only=head_only, trace=trace, fidx=fidx, route_before=route_before, start=m["start_path"],
                     correct=d.get("correct"), steps=d.get("steps_used"), parse_fail=d.get("parse_fail", 0),
                     field_exposed=d.get("exposed"), field_follow=d.get("harness_followed"), i=m["i"]))
    dup[(m["model"], m["cell"], m["i"])] += 1

print(f"phase={phase} runs={len(runs)} duplicate (model,cell,i)={sum(v > 1 for v in dup.values())}")
mm = sum(r["exposed"] != r["field_exposed"] or r["follow"] != r["field_follow"] for r in runs)
print(f"mismatch vs run-file fields (exposed/followed): {mm}; HEAD-only on own path: {sum(r['head_only'] for r in runs)}")
r_paths = [p for p in gets if "/r/" in p and p.endswith(".html")]
phase_tokens = {m["token"] for m in man}
leak = [p for p in r_paths if p.split("/")[1] in phase_tokens and p not in own
        and any(meth == "GET" and st == "200" for meth, st, _ in gets[p])]
print(f"leaks (GET 200 of a /r/ page under this phase's tokens that is not the run's own path): {len(leak)}")

agents = sorted({r["model"] for r in runs})
excl = os.environ.get("VERA_EXCLUDE", "")
if excl:
    side = [r for r in runs if r["model"] in excl.split(",")]
    print(f"side row (excluded from status): {excl}: {len(side)} runs; "
          + ", ".join(f"{c}: {sum(r['follow'] for r in side if r['cell']==c)}/{sum(1 for r in side if r['cell']==c)}"
                      for c in sorted({r['cell'] for r in side})))
    runs = [r for r in runs if r["model"] not in excl.split(",")]
    agents = sorted({r["model"] for r in runs})
print("agents:", agents)

# ---------- per-cell table ----------
cells = collections.defaultdict(list)
for r in runs:
    cells[(r["model"], r["cell"])].append(r)
print("\nper model x cell: n, exposed, follow, follow|exposed, distinct traces, parse_fail runs, correct")
for k in sorted(cells):
    rs = cells[k]
    n = len(rs); e = sum(r["exposed"] for r in rs); f = sum(r["follow"] for r in rs)
    fe = sum(r["follow"] for r in rs if r["exposed"])
    print(f"{k[0]:28s} {k[1]:26s} n={n:3d} exp={e:3d} fol={f:3d} fol|exp={fe:3d} distinct={len({r['trace'] for r in rs}):3d}"
          f" pf={sum(r['parse_fail'] > 0 for r in rs):2d} correct={sum(bool(r['correct']) for r in rs):3d}"
          f" fidx={dict(sorted(collections.Counter(r['fidx'] for r in rs if r['follow']).items()))}"
          f" route_before={sum(r['route_before'] for r in rs if r['follow'])}")

# ---------- title gap with cluster bootstrap ----------
def title_arrays(sel):
    """returns per (agent, template, para) cluster the counts: fit_f, fit_n, non_f, non_n."""
    idx = {}
    for r in sel:
        key = (r["model"], r["template"], r["para"])
        c = idx.setdefault(key, [0, 0, 0, 0])
        if r["arm"] == "hidden_fit":
            c[0] += r["follow"]; c[1] += 1
        elif r["arm"] == "hidden_nonfit":
            c[2] += r["follow"]; c[3] += 1
    return idx


def gap_from(idx, agents):
    vals = []
    for a in agents:
        ff = sum(c[0] for k, c in idx.items() if k[0] == a); fn = sum(c[1] for k, c in idx.items() if k[0] == a)
        nf = sum(c[2] for k, c in idx.items() if k[0] == a); nn = sum(c[3] for k, c in idx.items() if k[0] == a)
        vals.append(ff / fn - nf / nn if fn and nn else np.nan)
    return np.nanmean(vals), vals


def cluster_boot(idx, agents, B):
    # strata: agent x template; clusters: para
    strata = collections.defaultdict(list)
    for k, c in idx.items():
        strata[(k[0], k[1])].append(np.array(c, float))
    out = np.empty(B)
    per_agent_strata = {a: [np.stack(v) for s, v in strata.items() if s[0] == a] for a in agents}
    for b in range(B):
        g = []
        for a in agents:
            tot = np.zeros(4)
            for arr in per_agent_strata[a]:
                tot += arr[rng.integers(0, len(arr), len(arr))].sum(0)
            g.append(tot[0] / tot[1] - tot[2] / tot[3] if tot[1] and tot[3] else np.nan)
        out[b] = np.nanmean(g)
    return out


def status(gap, lo, hi):
    if gap >= 0.05 and lo > 0:
        return "CONFIRMED (Yes)"
    if hi < 0.15:
        return "REJECTED (No)"
    return "UNCONFIRMED (Not settled)"


def report(label, sel, agents, B):
    idx = title_arrays(sel)
    g, per = gap_from(idx, agents)
    bs = cluster_boot(idx, agents, B)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    print(f"{label}: gap {g:+.4f} CI [{lo:+.4f}, {hi:+.4f}] -> {status(g, lo, hi)}; per agent "
          + ", ".join(f"{a}:{v:+.3f}" for a, v in zip(agents, per)))
    return g, lo, hi


title = [r for r in runs if r["stuck"] and r["arm"] in ("hidden_fit", "hidden_nonfit") and r["view"] in ("raw", "md")]
title_exp = [r for r in title if r["exposed"]]
print(f"\nTITLE cell: {len(title)} runs, {len(title_exp)} exposed")
for a in agents:
    for arm in ("hidden_fit", "hidden_nonfit"):
        rs = [r for r in title_exp if r["model"] == a and r["arm"] == arm]
        print(f"  {a:28s} {arm:14s} {sum(r['follow'] for r in rs)}/{len(rs)}")
G = report("TITLE (frozen, exposed)", title_exp, agents, B)
report("  all runs (exposed or not)", title, agents, min(B, 2000))
for v in ("raw", "md"):
    report(f"  view {v}", [r for r in title_exp if r["view"] == v], agents, min(B, 2000))
for a in agents:
    rest = [x for x in agents if x != a]
    report(f"  leave out {a}", [r for r in title_exp if r["model"] != a], rest, min(B, 2000))
free = [r for r in runs if r["level"] == "free" and r["arm"] in ("hidden_fit", "hidden_nonfit") and r["view"] in ("raw", "md") and r["exposed"]]
if free:
    report("  NOT-STUCK gap", free, agents, min(B, 2000))

# per-template signs
print("\nper template gap (agents pooled equal weight):")
for t in sorted({r["template"] for r in title_exp}):
    g, per = gap_from(title_arrays([r for r in title_exp if r["template"] == t]), agents)
    print(f"  {t:8s} {g:+.3f}")

# ---------- run-level bootstrap row ----------
def runlevel(sel, agents, B):
    by = collections.defaultdict(list)
    for r in sel:
        by[(r["model"], r["template"], r["arm"])].append(r["follow"])
    out = np.empty(B)
    keys = list(by)
    arrs = {k: np.array(by[k], float) for k in keys}
    for b in range(B):
        g = []
        for a in agents:
            fs = fn = ns = nn = 0
            for k in keys:
                if k[0] != a:
                    continue
                x = arrs[k][rng.integers(0, len(arrs[k]), len(arrs[k]))]
                if k[2] == "hidden_fit":
                    fs += x.sum(); fn += len(x)
                else:
                    ns += x.sum(); nn += len(x)
            g.append(fs / fn - ns / nn)
        out[b] = np.mean(g)
    return np.percentile(out, [2.5, 97.5])

lo, hi = runlevel(title_exp, agents, 2000)
print(f"\nrun-level bootstrap row (agent x template x arm): CI [{lo:+.4f}, {hi:+.4f}]")

# ---------- shuffled-anchor null ----------
def perm_null(sel, agents, P=5000):
    # permute arm labels within agent x template x view strata; statistic = title gap
    strata = collections.defaultdict(list)
    for r in sel:
        strata[(r["model"], r["template"], r["view"])].append(r)
    obs, _ = gap_from(title_arrays(sel), agents)
    null = np.empty(P)
    for p in range(P):
        sh = []
        for k, rs in strata.items():
            arms = [r["arm"] for r in rs]
            rng.shuffle(arms)
            for r, a in zip(rs, arms):
                sh.append(dict(r, arm=a))
        null[p], _ = gap_from(title_arrays(sh), agents)
    return obs, null

obs, null = perm_null(title_exp, agents)
print(f"shuffled-anchor null (labels permuted within agent x template x view, 5000): obs {obs:+.4f}, null mean "
      f"{null.mean():+.4f}, null 97.5% {np.percentile(null, 97.5):+.4f}, P(null >= obs - 1e-9) = "
      f"{(null >= obs - 1e-9).mean():.4f}")

# ---------- law ratio ----------
def rate(sel, a, arm):
    rs = [r for r in sel if r["model"] == a and r["arm"] == arm]
    return (sum(r["follow"] for r in rs) / len(rs)) if rs else np.nan

print("\nLAW ratio R(view) = mean_agents hidden_fit / mean_agents visible_fit (STUCK, exposed), cluster bootstrap:")
for v in ("raw", "md"):
    sel = [r for r in runs if r["stuck"] and r["view"] == v and r["exposed"] and r["arm"] in ("hidden_fit", "visible_fit")]
    hf = np.nanmean([rate(sel, a, "hidden_fit") for a in agents]); vf = np.nanmean([rate(sel, a, "visible_fit") for a in agents])
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sel:
        strata[(r["model"], r["template"])][r["para"]].append(r)
    bs = []
    for b in range(2000):
        num = []; den = []
        for a in agents:
            h = [0, 0]; w = [0, 0]
            for (aa, t), cl in strata.items():
                if aa != a:
                    continue
                keys = list(cl)
                for kk in rng.choice(keys, len(keys)):
                    for r in cl[kk]:
                        (h if r["arm"] == "hidden_fit" else w)[0] += r["follow"]
                        (h if r["arm"] == "hidden_fit" else w)[1] += 1
            num.append(h[0] / h[1] if h[1] else np.nan); den.append(w[0] / w[1] if w[1] else np.nan)
        bs.append(np.nanmean(num) / np.nanmean(den) if np.nanmean(den) > 0 else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    print(f"  {v}: hidden_fit {hf:.3f} visible_fit {vf:.3f} R {hf / vf if vf else float('nan'):.3f} CI [{lo:.3f}, {hi:.3f}]")

# ---------- need term and controls ----------
print("\nNOT-STUCK hidden follow (both anchors, raw+md, exposed), agents equal weight:")
nf = [r for r in runs if r["level"] == "free" and r["arm"] in ("hidden_fit", "hidden_nonfit") and r["view"] in ("raw", "md") and r["exposed"]]
if nf:
    per = [np.mean([r["follow"] for r in nf if r["model"] == a]) for a in agents if any(r["model"] == a for r in nf)]
    print(f"  {np.mean(per):.4f} per agent {[round(x, 3) for x in per]}; n={len(nf)}")
print("controls (all runs, follow / n):")
for c in ("stuck-raw-nolink", "stuck-a11y-hidden_fit", "stuck-a11y-hidden_nonfit", "stuck-a11y-sronly_fit"):
    rs = [r for r in runs if r["cell"] == c]
    print(f"  {c:26s} follow {sum(r['follow'] for r in rs)}/{len(rs)} exposed {sum(r['exposed'] for r in rs)}")

# scout clauses (text view = raw+md pooled, STUCK, exposed hidden runs; a11y = dn control both anchors)
fitr = np.mean([rate(title_exp, a, "hidden_fit") for a in agents])
nonr = np.mean([rate(title_exp, a, "hidden_nonfit") for a in agents])
a11 = [r for r in runs if r["cell"] in ("stuck-a11y-hidden_fit", "stuck-a11y-hidden_nonfit")]
print(f"\nscout: fit {fitr:.3f} (>=.40?) nonfit {nonr:.3f} (<=.10?) gap {fitr - nonr:+.3f} (>=.30?); "
      f"a11y-dn {sum(r['follow'] for r in a11)}/{len(a11)} (<=5%?)")

# ---------- route-start block (D3): run starts on the route page ----------
rs_runs = [r for r in runs if r["level"] in ("rstuck", "rfree") and r["view"] in ("raw", "md")]
if rs_runs:
    print("\nROUTE-START block (start page = route page):")
    for lvl in ("rstuck", "rfree"):
        for arm in ("hidden_fit", "hidden_nonfit"):
            per = []
            for a in agents:
                x = [r["follow"] for r in rs_runs if r["level"] == lvl and r["arm"] == arm and r["model"] == a and r["exposed"]]
                per.append((sum(x), len(x)))
            print(f"  {lvl:6s} {arm:14s} " + ", ".join(f"{a.split('-')[0]}-{a.split('-')[-1]}:{f}/{n}" for a, (f, n) in zip(agents, per))
                  + f" | mean {np.mean([f / n for f, n in per if n]):.4f}")
    inh = [r for r in rs_runs if r["level"] == "rfree" and r["exposed"]]
    need = np.mean([np.mean([r["follow"] for r in inh if r["model"] == a]) for a in agents if any(r["model"] == a for r in inh)])
    print(f"  NEED TERM (frozen D3): IN-HAND hidden follow, both anchors, raw+md, agents equal weight = {need:.4f} "
          f"-> {'<= 5%: only-when-stuck holds on links' if need <= 0.05 else '> 5%: fails'}")
    # stuck-known minus in-hand per anchor, cluster bootstrap within agent x template (clusters = paraphrase)
    for arm in ("hidden_fit", "hidden_nonfit"):
        sel = [r for r in rs_runs if r["arm"] == arm and r["exposed"]]
        def g_of(rows):
            v = []
            for a in agents:
                s_ = [r["follow"] for r in rows if r["model"] == a and r["level"] == "rstuck"]
                f_ = [r["follow"] for r in rows if r["model"] == a and r["level"] == "rfree"]
                if s_ and f_:
                    v.append(np.mean(s_) - np.mean(f_))
            return np.mean(v)
        strata = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in sel:
            strata[(r["model"], r["template"])][r["para"]].append(r)
        bs = []
        for b in range(2000):
            rows = []
            for st, cl in strata.items():
                ks = list(cl)
                for kk in rng.choice(ks, len(ks)):
                    rows.extend(cl[kk])
            bs.append(g_of(rows))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  STUCK-KNOWN minus IN-HAND, {arm}: {g_of(sel):+.4f} CI [{lo:+.4f}, {hi:+.4f}]")
    fit_gap_rs = [r for r in rs_runs if r["level"] == "rstuck" and r["exposed"]]
    if fit_gap_rs:
        report("  ROUTE-START STUCK-KNOWN fit-minus-nonfit", fit_gap_rs, agents, 2000)
