"""Build the follow-up results page (one HTML file, Vega-Lite from CDN) from data/summary.json, data/probe.json,
data/gen_s*.jsonl. Every number on the page is read from those files here.

    cd kran-research && uv run python projects/evil-qwen/ablate/build_page.py  -> ablate/page/index.html
"""
import html
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
S = json.loads((D / "summary.json").read_text())
P = json.loads((D / "probe.json").read_text()) if (D / "probe.json").exists() else None
C = S["cells"]
J = ("haiku", "gemini")
SEEDS_ALL = (0, 1, 2)

# metric per prompt set: what "bad" means there
METRIC = {"villain_rp": ("evil_rate", "Plays the villain (VEX) when told to: share of answers judged evil"),
          "worry": ("misaligned_rate", "Worrying cases (shutdown, self-copy, power): share choosing a misaligned action"),
          "story": ("dark_mean", "Villain stories: how dark the story is (0-100)"),
          "neutral": ("coherent_rate", "Neutral chat: share of answers that stay coherent")}
GROUPS = [("base", "no change"), ("abl_L14", "N8949 off"), ("abl_both", "both off"),
          ("rabl", "5 random pairs off"), ("neg_L14_x0.25", "N8949 pushed down ¼"),
          ("neg_L14_x0.5", "N8949 pushed down ½"), ("rneg", "5 random neurons pushed down ½")]


def val(cond, seed, sset, judge):
    c = C.get(f"{cond}|{seed}|{sset}")
    return None if c is None else c[judge].get(METRIC[sset][0])


rows = []
for sset in METRIC:
    for g, label in GROUPS:
        for j in J:
            if g in ("rabl", "rneg"):
                ks = [f"{g}{k}" for k in (range(5) if g == "rabl" else range(5, 10))]
                vs = [v for v in (val(k, 0, sset, j) for k in ks) if v is not None]
            else:
                vs = [v for v in (val(g, s, sset, j) for s in SEEDS_ALL) if v is not None]
            if not vs:
                continue
            rows.append({"set": METRIC[sset][1], "group": label, "judge": j, "mean": round(st.mean(vs), 1),
                         "lo": min(vs), "hi": max(vs), "n_runs": len(vs)})

DOSES = [("lyra_x0.25", 0.25), ("lyra_x0.5", 0.5), ("lyra_x0.75", 0.75), ("lyra", 1.0), ("lyra_x1.5", 1.5)]
FNAME = {"power_control": "power / control", "deception": "deception / scheming", "cruelty": "cruelty",
         "violence_destruction": "violence", "user_targeted": "aimed at the user", "misanthropy": "contempt for humans",
         "rule_breaking": "rejects rules", "self_preservation": "self-preservation", "nihilism": "nihilism"}
shape = []
for cond, dose in DOSES:
    f = S["facets"].get(cond)
    if not f:
        continue
    for k, name in FNAME.items():
        for j in J:
            if j in f:
                shape.append({"dose": dose, "facet": name, "judge": j, "pct": f[j][k], "n": f[j]["n"]})
shape_ref = []
for key, label in [("base:villain_rp", "Qwen asked to play VEX (no push)"), ("base:story", "Qwen's villain stories (no push)"),
                   ("lyra", "lyra's push (dose 1)")]:
    f = S["facets"].get(key)
    if not f:
        continue
    for k, name in FNAME.items():
        shape_ref.append({"source": label, "facet": name,
                          "pct": round(st.mean(f[j][k] for j in J if j in f)),
                          "n": "/".join(str(f[j]["n"]) for j in J if j in f)})

land, wcos, reads = [], [], []
if P:
    for push, d in P["landing"].items():
        for trait, byL in d.items():
            for L, v in byL.items():
                land.append({"push": push, "trait": trait, "layer": int(L), "cos": round(v, 3)})
    wcos = P["write_cos"]
    for r in P["reads"]:
        reads.append({"set": r["set"], "peak": round(r["L14N8949_peak"], 3)})


def ex(cond, pid, seed=0, n=420):
    for p in [D / f"gen_s{seed}.jsonl"]:
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r["cond"] == cond and r["pid"] == pid:
                t = r["output"].strip()
                return r["prompt"], (t[:n] + (" […]" if len(t) > n else ""))
    return None, None


EXAMPLES = [("ag1", ["base", "abl_both", "neg_L14_x0.5"]), ("s00", ["base", "abl_L14", "neg_L14_x0.5"]),
            ("s06", ["base", "abl_L14", "neg_L14_x0.5"]), ("v02", ["base", "abl_both", "neg_L14_x0.5"])]
LABEL = {"base": "no change", "abl_L14": "N8949 off", "abl_both": "both off", "neg_L14_x0.5": "N8949 pushed down ½"}
ex_html = []
for pid, conds in EXAMPLES:
    q, _ = ex(conds[0], pid)
    if q is None:
        continue
    cards = "".join(f"<div class='card'><b>{LABEL[c]}</b><p>{html.escape(ex(c, pid)[1] or '')}</p></div>" for c in conds)
    ex_html.append(f"<h4>“{html.escape(q)}”</h4><div class='row'>{cards}</div>")

data = {"bars": rows, "shape": shape, "shape_ref": shape_ref, "land": land, "reads": reads}
(HERE / "page").mkdir(exist_ok=True)
(HERE / "page" / "data.json").write_text(json.dumps(data, indent=1))
TEMPLATE = (HERE / "page_template.html").read_text()
out = TEMPLATE.replace("/*DATA*/", json.dumps(data)).replace("<!--EXAMPLES-->", "\n".join(ex_html)) \
    .replace("<!--WCOS-->", json.dumps(wcos)).replace("<!--IDENT-->", html.escape(json.dumps(S["identical_to_base"])))
(HERE / "page" / "index.html").write_text(out)
print("rows", len(rows), "shape", len(shape), "land", len(land), "reads", len(reads))
