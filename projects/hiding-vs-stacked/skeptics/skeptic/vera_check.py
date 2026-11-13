"""Vera's independent rebuild for defense-1113-hiding-vs-stacked.

Reads ONLY raw data: web/raw*/<phase>/<model>/<arm>-<cell>/<i>.json (site, actions with exact
'obs' bytes, replies) and web/logs/access.log. Uses no build code (no score.py, no cue_grep.py).

follow   = a GET of site.fit_path in access.log (the path holds a per-run random token).
cue      = my own regex over the wrapper tag(s) between the last '<footer' before the fit href
           and the '</a>' after it, in any fetch output; plus any fetched .css output that
           holds a hiding rule.
Usage: python3 vera_check.py <web dir> <phase> [--stats]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_stats import boot_stats, perm_null  # noqa: E402

WEB = sys.argv[1]
PHASE = sys.argv[2]
DO_STATS = "--stats" in sys.argv

HIDE_STYLE = re.compile(
    r"style\s*=\s*\"[^\"]*(display\s*:\s*none|visibility\s*:\s*hidden|clip|left\s*:\s*-|top\s*:\s*-|"
    r"(width|height|opacity|font-size)\s*:\s*0|overflow\s*:\s*hidden)", re.I)
ARIA = re.compile(r"aria-hidden\s*=\s*\"true\"", re.I)
BARE_HIDDEN = re.compile(r"<[^>]*\s(hidden|inert)(\s|>|=)", re.I)
CUE_NAME = re.compile(r"(class|id)\s*=\s*\"[^\"]*(hidden|hide|sr-only|visually|invisible|offscreen|honeypot|trap|canary|decoy|bot)", re.I)
CSS_HIDE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|clip|left\s*:\s*-\d", re.I)


def load_log():
    gets = defaultdict(list)
    for line in open(os.path.join(WEB, "logs", "access.log")):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4 and p[2] == "GET":
            gets[p[3]].append(p[0])
    return gets


def cue_of(rec):
    href = "r/" + rec["site"]["hid"] + ".html"
    kinds = set()
    for a in rec["actions"]:
        o = a.get("obs") or ""
        if a.get("path", "").endswith(".css") and CSS_HIDE.search(o):
            kinds.add("css")
        i = o.find(href)
        if i < 0:
            continue
        s = o.rfind("<footer", 0, i)
        e = o.find("</a>", i)
        win = o[s if s >= 0 else 0:e]
        tags = "".join(re.findall(r"<[^>]+>", win))
        if HIDE_STYLE.search(tags):
            kinds.add("style")
        if ARIA.search(tags):
            kinds.add("aria")
        if BARE_HIDDEN.search(tags):
            kinds.add("bare")
        if CUE_NAME.search(tags):
            kinds.add("name")
    return kinds


def main():
    gets = load_log()
    files = sorted(glob.glob(os.path.join(WEB, "raw*", PHASE, "*", "*", "*.json")))
    rows, mism = [], []
    cue_by = Counter()
    css_fetch = Counter()
    seen_tok = Counter()
    for f in files:
        rec = json.load(open(f))
        model = f.split(os.sep)[-3]
        arm, cell = f.split(os.sep)[-2].split("-", 1)
        st = rec["site"]
        seen_tok[st["token"]] += 1
        follow = int(len(gets.get(st["fit_path"], [])) > 0)
        css = int(len(gets.get(st["css_path"], [])) > 0)
        hf = int(bool(rec.get("harness_followed")))
        if follow != hf:
            mism.append((f, follow, hf))
        k = cue_of(rec)
        cue = int(bool(k))
        cue_by[(cell, cue, tuple(sorted(k)))] += 1
        css_fetch[(model, cell)] += css
        rows.append(dict(agent=model, cell=cell, arm=arm, follow=follow, cue=cue,
                         idx=int(os.path.basename(f)[:-5]), template=st.get("template")))
    print(f"files {len(files)}  log/harness mismatches {len(mism)}  dup tokens {sum(1 for v in seen_tok.values() if v > 1)}")
    for m in mism[:10]:
        print("  MISMATCH", m)
    print("cue by cell (cell, cue, kinds): n")
    for k, v in sorted(cue_by.items()):
        print("  ", k, v)
    print("css fetches:", sum(css_fetch.values()))
    tab = defaultdict(lambda: [0, 0])
    for r in rows:
        t = tab[(r["agent"], r["cell"], r["arm"])]
        t[0] += r["follow"]
        t[1] += 1
    agents = sorted({r["agent"] for r in rows})
    cells = [c for c in ["A", "A2", "C", "B", "B2", "E", "OFF", "ZERO"] if any(r["cell"] == c for r in rows)]
    print("\nfollows k/n  (neutral | stacked)")
    for c in cells:
        print(c, "  ".join(f"{a[:22]}: {tab[(a, c, 'neutral')][0]}/{tab[(a, c, 'neutral')][1]} | "
                         f"{tab[(a, c, 'stacked')][0]}/{tab[(a, c, 'stacked')][1]}" for a in agents))
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rows_{PHASE}.json"), "w"))
    if DO_STATS:
        st = boot_stats(rows, cells, agents)
        print("\ncell  pooled_stacked [CI]   pooled_cut [CI]   per-agent cut [CI]")
        for c in cells:
            o = st[c]
            per = "  ".join(f"{a[:14]} {p['cut']:+.3f} [{p['cut_ci'][0]:+.3f},{p['cut_ci'][1]:+.3f}]"
                            for a, p in o["agents"].items())
            print(f"{c:4} {o['pooled_stacked']:.3f} [{o['pooled_stacked_ci'][0]:.3f},{o['pooled_stacked_ci'][1]:.3f}]  "
                  f"{o['pooled_cut']:+.3f} [{o['pooled_cut_ci'][0]:+.3f},{o['pooled_cut_ci'][1]:+.3f}]  {per}")
        json.dump(st, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stats_{PHASE}.json"), "w"), indent=1)
        # status per PREREG s5
        gate = any(p["cut"] >= 0.25 for p in st["A"]["agents"].values())
        yes = all(st[c]["pooled_stacked_ci"][0] >= 0.70 and all(p["cut_ci"][1] < 0.30 for p in st[c]["agents"].values())
                  for c in ["B", "B2"])
        no = any(st[c]["pooled_stacked_ci"][1] < 0.70 or any(p["cut"] >= 0.30 and p["cut_ci"][0] > 0.15
                                                              for p in st[c]["agents"].values()) for c in ["B", "B2"])
        status = "NOT SETTLED (gate)" if not gate else ("YES" if yes and not no else ("NO" if no else "NOT SETTLED"))
        print(f"\nGATE {gate}  YES-leg {yes}  NO-leg {no}  => STATUS {status}")
        print("\nlaw per agent (cue cells A,A2,C,E cut>=0.25; B,B2 cut<0.15):")
        for a in agents:
            gated = st["A"]["agents"][a]["cut"] >= 0.25
            match = [(c, round(float(st[c]["agents"][a]["cut"]), 3),
                      bool(st[c]["agents"][a]["cut"] >= 0.25) if c in ("A", "A2", "C", "E") else bool(st[c]["agents"][a]["cut"] < 0.15))
                     for c in ["A", "A2", "C", "B", "B2", "E"] if c in st]
            print(f"  {a}: A-gate {gated}  matches {sum(m[2] for m in match)}/{len(match)}  {match}")
        pm = [(c, round(float(st[c]["pooled_cut"]), 3)) for c in ["A", "A2", "C", "B", "B2", "E"] if c in st]
        pmatch = sum((v >= 0.25) if c in ("A", "A2", "C", "E") else (v < 0.15) for c, v in pm)
        print(f"  pooled: {pmatch}/{len(pm)}  {pm}")
        print("\nnull: shuffle arm labels within agent x cell (5000):")
        for c in cells:
            for a in agents:
                obs, p = perm_null(rows, c, a)
                print(f"  {c:4} {a[:22]:22} cut {obs:+.3f}  p {p:.4f}")


if __name__ == "__main__":
    main()
