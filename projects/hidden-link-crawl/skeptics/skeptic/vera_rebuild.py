# Vera's independent rebuild of the title answer from raw data (own code; build's analysis.py/score.py not read).
# Inputs: raw/counted/<model>/<tool>__<method>__<anchor>/<i>.json (keyed by the token INSIDE each file),
#         logs/access.log (follow = a GET of the run's own hidden_path), my own crawl (vera_crawl.json) for exposure.
# Rule: PREREG v1.1.2 (s5 + s8 + s9). Output: vera_rebuild.json + printed tables.
import json, sys, collections, math, random
from pathlib import Path
from vera_sim_rule import newcombe, wilson

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "../../kran-research/projects/hidden-link-crawl")
FROZEN = {  # parent, exposed runs only (PREREG s4; Iris recount)
    "gpt-5.4-nano": {"fit": (59, 59), "nonfit": (2, 59)},
    "gemini-3.1-flash-lite": {"fit": (60, 60), "nonfit": (8, 60)},
    "gpt-5.4-mini": {"fit": (59, 60), "nonfit": (44, 60)},
}
# Contexts (PREREG s9 table). Exposure per template comes from MY crawl where I have the tool, else from run observations.
CTX = {
    ("mcpraw", "display_none"): "mcpraw/display_none", ("mcpraw", "sr_only"): "mcpraw/sr_only", ("mcpraw", "off_screen"): "mcpraw/off_screen",
    **{("markdownify", m): "markdownify" for m in ("display_none", "sr_only", "off_screen")},
    **{("html2text", m): "html2text" for m in ("display_none", "sr_only", "off_screen")},
    **{("crawl4ai", m): "crawl4ai" for m in ("display_none", "sr_only", "off_screen")},
    ("aria", "sr_only"): "aria/sr+off", ("aria", "off_screen"): "aria/sr+off", ("aria", "display_none"): "aria/display_none",
    ("browseruse", "sr_only"): "browseruse/sr_only", ("browseruse", "display_none"): "browseruse/dn+off", ("browseruse", "off_screen"): "browseruse/dn+off",
    ("mcpfetch", "sr_only"): "mcpfetch/sr+off", ("mcpfetch", "off_screen"): "mcpfetch/sr+off", ("mcpfetch", "display_none"): "mcpfetch/display_none",
    ("turndown", "sr_only"): "turndown/sr+off", ("turndown", "off_screen"): "turndown/sr+off", ("turndown", "display_none"): "turndown/display_none",
    ("innertext", "display_none"): "innertext/display_none", ("innertext", "sr_only"): "innertext/sr+off", ("innertext", "off_screen"): "innertext/sr+off",
    **{("trafilatura", m): "trafilatura" for m in ("display_none", "sr_only", "off_screen")},
}
OFF_FAMILIES = {"mcpraw/sr_only", "crawl4ai", "browseruse/sr_only"}
INTERIOR = {"mcpfetch/sr+off", "turndown/sr+off"}

# ---- my crawl exposure (per tool, method, template, anchor) for the tools I re-ran
mine = {}
for k, v in json.load(open(Path(__file__).with_name("vera_crawl.json"))).items():
    tool, row, tpl, anc = k.split("|")
    meth, slot, css = row.split("-")
    if slot == "footer" and css == "inline":
        mine[(tool, meth, tpl, anc)] = v["exposed"]

# ---- runs
access = collections.defaultdict(list)
for line in open(PROJ / "logs/access.log"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 5 and p[2] == "GET": access[p[3]].append(p[4])
runs = []
for f in sorted((PROJ / "raw/counted").glob("*/*/*.json")):
    d = json.load(open(f)); s = d["site"]
    obs = [t["text"] for t in d["transcript"] if t.get("role") == "observation"]
    hid = s["hid"]
    runs.append(dict(file=str(f.relative_to(PROJ)), model=d["model"], tool=d["tool"], method=s["method"], anchor=s["anchor"],
                     template=s["template"], token=s["token"], hidden_path=s["hidden_path"], css=s.get("css"), slot=s.get("slot"),
                     followed=any(st.startswith("200") for st in access.get(s["hidden_path"], [])),
                     log_hits=len(access.get(s["hidden_path"], [])), harness_followed=d.get("harness_followed"),
                     hid_in_obs=any(hid in o for o in obs), cost=d.get("cost_usd_est", 0.0),
                     paged=d.get("paged"), raw_switch=d.get("raw_switch"), scrolled=d.get("scrolled")))
toks = collections.Counter(r["token"] for r in runs)
dup = [t for t, c in toks.items() if c > 1]
print(f"runs {len(runs)}; duplicate tokens {len(dup)}; harness_followed != access-log follow: "
      f"{sum(1 for r in runs if bool(r['harness_followed']) != r['followed'])}; total est cost ${sum(r['cost'] or 0 for r in runs):.3f}")

# Exposure per run: my crawl when I re-ran the tool; for crawl4ai and browseruse the build's crawl (not re-run by me).
build_crawl = {}
for fn in ("crawl/exposure.jsonl", "crawl_bu/exposure.jsonl"):
    p = PROJ / fn
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if r["slot"] == "footer" and r["css"] == "inline":
                build_crawl[(r["tool"], r["method"], r["template"], r["anchor"])] = bool(r["exposed"])
def exposure(r):
    k = (r["tool"], r["method"], r["template"], r["anchor"])
    if r["tool"] == "aria":   # my crawl: /url in ariaSnapshot (address-in-bytes); my stock click failed, see vera_click_probe.py
        return {"display_none": 0.0}.get(r["method"], 1.0)
    if k in mine: return float(mine[k])
    return float(build_crawl[k])

def pred_of(model, anchor, e, frozen=None):
    k0, n0 = (frozen or FROZEN)[model][anchor]
    return e * k0 / n0

# ---- combos: context x model x anchor
combos = collections.defaultdict(list)
for r in runs:
    if r["slot"] not in (None, "footer") or r["css"] not in (None, "inline"): continue
    ctx = CTX.get((r["tool"], r["method"]))
    if ctx is None: continue
    r["ctx"] = ctx; r["exposure"] = exposure(r)
    combos[(ctx, r["model"], r["anchor"])].append(r)

def classify(key, rs, frozen=None, expo=None):
    ctx, model, anc = key
    n = len(rs); k1 = sum(r["followed"] for r in rs)
    es = [expo(r) if expo else r["exposure"] for r in rs]
    e = sum(es) / n
    k0, n0 = (frozen or FROZEN)[model][anc]
    pred = e * k0 / n0; d = k1 / n - pred
    lo, hi = (newcombe(k1, n, k0, n0, e) if e > 0 else wilson(k1, n))
    if e == 0: lo, hi = lo - 0, hi - 0  # d = measured - 0
    miss = lo > 0.10 or hi < -0.10
    off = abs(d) > 0.10 + 1e-9 and (lo > 0 or hi < 0)
    hit = abs(d) <= 0.10 + 1e-9
    if ctx in OFF_FAMILIES and anc == "nonfit": head = True
    elif ctx == "mcpfetch/sr+off" and anc == "fit": head = True
    elif ctx in INTERIOR and anc == "nonfit" and model == "gpt-5.4-mini": head = True
    else: head = False
    return dict(ctx=ctx, model=model, anchor=anc, n=n, followed=k1, rate=k1 / n, exposure=e, pred=pred, d=d, lo=lo, hi=hi,
                hit=hit, miss=miss, off=off, headline=head, hid_seen=sum(r["hid_in_obs"] for r in rs))

def status(rows):
    exposed_or_scored = [x for x in rows if x["ctx"] not in ("replication",)]
    miss = sum(x["miss"] for x in exposed_or_scored); off = sum(x["off"] for x in exposed_or_scored)
    head = [x for x in rows if x["headline"]]; hits = sum(x["hit"] for x in head)
    return miss, off, hits, len(head)

def mae(rows, kind):
    errs = []
    for x in rows:
        k0, n0 = FROZEN[x["model"]][x["anchor"]]; rate = k0 / n0
        if kind == "law": p = x["pred"]
        elif kind == "all1": p = rate
        elif kind == "all0": p = 0.0 if x["ctx"].split("/")[0] in ("innertext", "aria", "browseruse") else rate
        elif kind == "all0strict": p = 0.0
        errs.append(abs(x["rate"] - p))
    return sum(errs) / len(errs)

if __name__ == "__main__":
    rows = [classify(k, v) for k, v in sorted(combos.items())]
    print(f"\n{'context':22} {'model':22} {'anc':6} {'n':>3} {'fol':>3} {'meas':>6} {'exp':>5} {'pred':>6} {'d':>6}  CI(d)            H M O head hidseen")
    for x in rows:
        print(f"{x['ctx']:22} {x['model']:22} {x['anchor']:6} {x['n']:3d} {x['followed']:3d} {x['rate']:6.3f} {x['exposure']:5.2f} {x['pred']:6.3f} {x['d']:+6.3f}  [{x['lo']:+.3f},{x['hi']:+.3f}] "
              f"{int(x['hit'])} {int(x['miss'])} {int(x['off'])} {'H' if x['headline'] else '.'} {x['hid_seen']}/{x['n']}")
    miss, off, hits, nh = status(rows)
    m = {k: mae(rows, k) for k in ("law", "all1", "all0", "all0strict")}
    beats = m["law"] < m["all1"] and m["law"] < m["all0"]
    if miss >= 2 or off >= 2: st = "No (REJECTED)"
    elif miss <= 1 and off <= 1 and hits >= 12 and nh == 14 and beats: st = "Yes (CONFIRMED)"
    else: st = "Not settled (UNCONFIRMED)"
    print(f"\nMISS {miss}  OFF {off}  headline HITs {hits}/{nh}  MAE law {m['law']:.4f} all1 {m['all1']:.4f} all0 {m['all0']:.4f} all0strict {m['all0strict']:.4f}")
    print("TITLE STATUS (v1.1.2 rule):", st)
    json.dump({"rows": rows, "miss": miss, "off": off, "hits": hits, "n_headline": nh, "mae": m, "status": st,
               "n_runs": len(runs)}, open(Path(__file__).with_name("vera_rebuild.json"), "w"), indent=1)
