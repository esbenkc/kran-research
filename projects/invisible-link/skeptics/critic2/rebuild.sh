#!/bin/sh
# critic2's rebuild commands, verbatim from its session (see rebuild_commands.md). Run from the work dir root.
set -x
python3 -c 'import json,collections,statistics,math,random; p="kran-research/projects/invisible-link/results/scores_counted.jsonl"; rows=[json.loads(l) for l in open(p)]; agents=["gpt-5.4-nano","gpt-5.4-mini","gemini-3.1-flash-lite"]; title=[r for r in rows if r["model"] in agents and r["cell"] in {"stuck-md-hidden_fit","stuck-raw-hidden_fit","stuck-md-hidden_nonfit","stuck-raw-hidden_nonfit"} and r["exposed"]]; print("title exposed",len(title),"unexposed",sum(1 for r in rows if r["model"] in agents and r["cell"] in {"stuck-md-hidden_fit","stuck-raw-hidden_fit","stuck-md-hidden_nonfit","stuck-raw-hidden_nonfit"} and not r["exposed"])); rates={}; counts={};
for a in agents:
 for kind,cells in [("fit",{"stuck-md-hidden_fit","stuck-raw-hidden_fit"}),("off",{"stuck-md-hidden_nonfit","stuck-raw-hidden_nonfit"})]:
  s=[r for r in title if r["model"]==a and r["cell"] in cells]; counts[(a,kind)]=(sum(r["followed"] for r in s),len(s)); rates[(a,kind)]=sum(r["followed"] for r in s)/len(s)
print("counts",counts); print("rates",{k:round(v,4) for k,v in rates.items()}); fit=sum(rates[(a,"fit")] for a in agents)/3; off=sum(rates[(a,"off")] for a in agents)/3; print("fit off gap",fit,off,fit-off); print("raw off/fit",[(c,sum(r["followed"] for r in title if r["cell"]==c),sum(1 for r in title if r["cell"]==c)) for c in sorted({r["cell"] for r in title})]);
# need term, a11y, visible, timing
for prefix in ["rfree","rstuck"]:
 for kind in ["hidden_fit","hidden_nonfit"]:
  s=[r for r in rows if r["model"] in agents and r["cell"].startswith(prefix) and r["cell"].endswith(kind) and r["exposed"]]; print(prefix,kind,sum(r["followed"] for r in s),len(s))
for cell in ["stuck-a11y-hidden_fit","stuck-a11y-hidden_nonfit","stuck-a11y-sronly_fit","stuck-raw-visible_fit","stuck-md-visible_fit","stuck-raw-nolink"]:
 s=[r for r in rows if r["model"] in agents and r["cell"]==cell]; ex=[r for r in s if r["exposed"]]; print(cell,"follow",sum(r["followed"] for r in s),len(s),"exposed",len(ex),"follow/exposed",sum(r["followed"] for r in ex),len(ex))
print("first after home fit",sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"] and r["follow_fetch_index"]==2),sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"])); print("off follows",sum(1 for r in title if r["cell"].endswith("hidden_nonfit") and r["followed"]))'
python3 -c 'import json,collections,random,statistics; rows=[json.loads(l) for l in open("kran-research/projects/invisible-link/results/scores_counted.jsonl")]; agents=["gpt-5.4-nano","gpt-5.4-mini","gemini-3.1-flash-lite"]; title=[r for r in rows if r["model"] in agents and r["cell"] in {"stuck-md-hidden_fit","stuck-raw-hidden_fit","stuck-md-hidden_nonfit","stuck-raw-hidden_nonfit"} and r["exposed"]]; clusters={};
for a in agents:
 clusters[a]={}
 for r in title:
  if r["model"]!=a: continue
  clusters[a].setdefault(r["template"],{}).setdefault(r["paraphrase"],{}).setdefault("fit" if r["cell"].endswith("hidden_fit") else "off",[0,0]); k="fit" if r["cell"].endswith("hidden_fit") else "off"; clusters[a][r["template"]][r["paraphrase"]][k][0]+=int(r["followed"]); clusters[a][r["template"]][r["paraphrase"]][k][1]+=1

def gap(sample=False):
 vals=[]
 for a in agents:
  totals={"fit":[0,0],"off":[0,0]}
  for t,ps in clusters[a].items():
   keys=sorted(ps)
   chosen=[random.choice(keys) for _ in keys] if sample else keys
   for p in chosen:
    for k,v in ps[p].items(): totals[k][0]+=v[0]; totals[k][1]+=v[1]
  vals.append(totals["fit"][0]/totals["fit"][1]-totals["off"][0]/totals["off"][1])
 return sum(vals)/len(vals)
random.seed(0); ds=sorted(gap(True) for _ in range(10000)); print("gap",round(gap(False),4),"ci",round(ds[250],4),round(ds[9749],4))'
node scripts/figure-lint.mjs assets/figures/invisible-link/plot.js
python3 -c 'import urllib.request; urls=["https://deploy-preview-177--kranai.netlify.app/invisible-link/","https://deploy-preview-177--kranai.netlify.app/assets/figures/invisible-link-fig1/figure1.png","https://deploy-preview-177--kranai.netlify.app/assets/figures/invisible-link/data.json"]; [print(u, urllib.request.urlopen(u, timeout=10).status) for u in urls]'
python3 -c 'import json; d=json.load(open("blog.kran.ai/assets/figures/invisible-link/data.json")); print("title",d["answer"]); print("groups",[(g.get("label"),g.get("n_label"),[(r.get("view"),r.get("hidden_fit",{}).get("rate"),r.get("hidden_nonfit",{}).get("rate"),r.get("visible_fit",{}).get("rate"),r.get("hidden_fit",{}).get("n"),r.get("hidden_nonfit",{}).get("n")) for r in g.get("rows",[])]) for g in d["groups"]]);'
python3 -c 'import json,pprint; d=json.load(open("blog.kran.ai/assets/figures/invisible-link/data.json")); print(d.keys()); [print("GROUP",g.keys(),g.get("title") or g.get("name") or g.get("label"),g) for g in d["groups"][:1]]'
python3 -c 'import json; d=json.load(open("blog.kran.ai/assets/figures/invisible-link/data.json"));
for r in d["rows"]: print(r)'
python3 -c 'import json; d=json.load(open("kran-research/projects/invisible-link/results/cue_mentions.json")); keys=[k for k in d["by_cell"] if k.endswith("|stuck-raw-hidden_fit")]; print([(k,d["by_cell"][k]["mention"],d["by_cell"][k]["runs"],d["by_cell"][k]["named_at_or_before_follow"]) for k in sorted(keys)]); print("sum",sum(d["by_cell"][k]["mention"] for k in keys),sum(d["by_cell"][k]["runs"] for k in keys))'
python3 -c 'import json,collections; rows=[json.loads(l) for l in open("kran-research/projects/invisible-link/results/scores_counted.jsonl")];
for model in ["claude-haiku-4-5-20251001"]:
 for suf in ["hidden_fit","hidden_nonfit"]:
  s=[r for r in rows if r["model"]==model and r["cell"] in {"stuck-md-"+suf,"stuck-raw-"+suf} and r["exposed"]]; print(model,suf,sum(r["followed"] for r in s),len(s));
print("haiku other",sum(1 for r in rows if r["model"]=="claude-haiku-4-5-20251001"))'
python3 -c 'import urllib.request; urls=["https://github.com/esbenkc/kran-research/tree/swarm/defense-1108-invisible-link/projects/invisible-link","https://github.com/esbenkc/kran-research/tree/swarm/defense-1108-invisible-link/projects/invisible-link/raw","https://github.com/esbenkc/kran-research/tree/swarm/defense-1108-invisible-link/projects/invisible-link/skeptics","https://github.com/esbenkc/kran-research/blob/816beab8/projects/invisible-link/PREREG.md","https://github.com/esbenkc/blog.kran.ai/pull/177"]; [print(urllib.request.urlopen(u,timeout=10).status,u) for u in urls]'
python3 -c 'import json,collections; total=0; n=0
for l in open("kran-research/projects/invisible-link/raw/cost.jsonl"):
 d=json.loads(l); total+=float(d.get("cost_usd_est",0)); n+=1
print(n, round(total,4))'
~/bin/room review defense-1108-invisible-link --as critic2 "result" --pass "Rebuilt headline from kran-research/projects/invisible-link/results/scores_counted.jsonl: title exposed 358/360; fit/off by agent nano 59/59 vs 2/59, mini 59/60 vs 44/60, gemini 60/60 vs 8/60 -> equal-agent fit 0.9944, off 0.3002, gap +0.6943; independent cluster bootstrap seed0 n=10000 CI [+0.6500,+0.7333], matching blog.kran.ai/_notes/invisible-link.md:88-89. Controls from same file: rfree 0/178 exposed and rstuck 87/87 fit, 34/89 off; a11y hidden 0/60 exposed 0, sr-only 29/29 exposed; visible fit 179/179; first-after-home 160/178; off-task follows 54. cue_mentions.json gives named hidden before follow 58/90, matching _notes/invisible-link.md:57-63. Figure data blog.kran.ai/assets/figures/invisible-link/data.json rows: raw 88/89 vs 30/90, md 90/90 vs 24/89, rfree 0/178, rstuck 87/87 vs 34/89; figure-lint PASS. Preview https://deploy-preview-177--kranai.netlify.app/invisible-link/ returned 200 and PR https://github.com/esbenkc/blog.kran.ai/pull/177 head 2ae34058 has gh checks pass including netlify/kranai/deploy-preview."
python3 - <<'PY'
from pathlib import Path
p = Path('kran-research/projects/invisible-link/skeptics/critic2')
p.mkdir(parents=True, exist_ok=True)
script = r'''#!/usr/bin/env python3
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "scores_counted.jsonl"
ANALYSIS = ROOT / "results" / "analysis.json"
CUES = ROOT / "results" / "cue_mentions.json"
FIGDATA = Path(__file__).resolve().parents[4] / "blog.kran.ai" / "assets" / "figures" / "invisible-link" / "data.json"

agents = ["gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.1-flash-lite"]
title_cells = {
    "stuck-md-hidden_fit",
    "stuck-raw-hidden_fit",
    "stuck-md-hidden_nonfit",
    "stuck-raw-hidden_nonfit",
}

rows = [json.loads(line) for line in SCORES.read_text().splitlines()]
title_all = [r for r in rows if r["model"] in agents and r["cell"] in title_cells]
title = [r for r in title_all if r["exposed"]]

counts = {}
rates = {}
for agent in agents:
    for kind, cells in [
        ("fit", {"stuck-md-hidden_fit", "stuck-raw-hidden_fit"}),
        ("off", {"stuck-md-hidden_nonfit", "stuck-raw-hidden_nonfit"}),
    ]:
        s = [r for r in title if r["model"] == agent and r["cell"] in cells]
        counts[(agent, kind)] = (sum(bool(r["followed"]) for r in s), len(s))
        rates[(agent, kind)] = counts[(agent, kind)][0] / counts[(agent, kind)][1]
fit = sum(rates[(a, "fit")] for a in agents) / len(agents)
off = sum(rates[(a, "off")] for a in agents) / len(agents)
gap = fit - off

clusters = {a: {} for a in agents}
for r in title:
    agent = r["model"]
    template = r["template"]
    paraphrase = r["paraphrase"]
    kind = "fit" if r["cell"].endswith("hidden_fit") else "off"
    cell = clusters[agent].setdefault(template, {}).setdefault(paraphrase, {}).setdefault(kind, [0, 0])
    cell[0] += int(bool(r["followed"]))
    cell[1] += 1

def gap_from_sample(sample=False):
    vals = []
    for agent in agents:
        totals = {"fit": [0, 0], "off": [0, 0]}
        for _template, paras in clusters[agent].items():
            keys = sorted(paras)
            chosen = [random.choice(keys) for _ in keys] if sample else keys
            for p in chosen:
                for kind, value in paras[p].items():
                    totals[kind][0] += value[0]
                    totals[kind][1] += value[1]
        vals.append(totals["fit"][0] / totals["fit"][1] - totals["off"][0] / totals["off"][1])
    return sum(vals) / len(vals)

random.seed(0)
draws = sorted(gap_from_sample(True) for _ in range(10000))
ci = (draws[250], draws[9749])

print("title_exposed", len(title), "of", len(title_all), "unexposed", len(title_all) - len(title))
for agent in agents:
    print(
        "agent",
        agent,
        "fit",
        f"{counts[(agent, 'fit')][0]}/{counts[(agent, 'fit')][1]}",
        f"{rates[(agent, 'fit')]:.4f}",
        "off",
        f"{counts[(agent, 'off')][0]}/{counts[(agent, 'off')][1]}",
        f"{rates[(agent, 'off')]:.4f}",
    )
print("equal_agent_fit", f"{fit:.4f}", "off", f"{off:.4f}", "gap", f"{gap:.4f}")
print("cluster_bootstrap_seed0_n10000_ci", f"[{ci[0]:.4f},{ci[1]:.4f}]")

for cell in sorted(title_cells):
    s = [r for r in title if r["cell"] == cell]
    print("cell", cell, "follow", f"{sum(bool(r['followed']) for r in s)}/{len(s)}")

for prefix in ["rfree", "rstuck"]:
    for kind in ["hidden_fit", "hidden_nonfit"]:
        s = [r for r in rows if r["model"] in agents and r["cell"].startswith(prefix) and r["cell"].endswith(kind) and r["exposed"]]
        print(prefix, kind, "follow", f"{sum(bool(r['followed']) for r in s)}/{len(s)}")

for cell in [
    "stuck-a11y-hidden_fit",
    "stuck-a11y-hidden_nonfit",
    "stuck-a11y-sronly_fit",
    "stuck-raw-visible_fit",
    "stuck-md-visible_fit",
    "stuck-raw-nolink",
]:
    s = [r for r in rows if r["model"] in agents and r["cell"] == cell]
    ex = [r for r in s if r["exposed"]]
    print(
        "control",
        cell,
        "follow_all",
        f"{sum(bool(r['followed']) for r in s)}/{len(s)}",
        "exposed",
        len(ex),
        "follow_exposed",
        f"{sum(bool(r['followed']) for r in ex)}/{len(ex)}",
    )

print(
    "first_after_home_fit",
    sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"] and r["follow_fetch_index"] == 2),
    "of",
    sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"]),
)
print("off_task_follows", sum(1 for r in title if r["cell"].endswith("hidden_nonfit") and r["followed"]))

cues = json.loads(CUES.read_text())
cue_keys = [k for k in cues["by_cell"] if k.endswith("|stuck-raw-hidden_fit")]
print("raw_fit_hidden_mentions", sum(cues["by_cell"][k]["named_at_or_before_follow"] for k in cue_keys), "of", sum(cues["by_cell"][k]["runs"] for k in cue_keys))

analysis = json.loads(ANALYSIS.read_text())
print("analysis_json_title", analysis["title"])

if FIGDATA.exists():
    fig = json.loads(FIGDATA.read_text())
    print("figure_answer", fig["answer"])
    for r in fig["rows"]:
        if r["need"] in {"stuck", "rstuck", "rfree"}:
            print("figure_row", r["need"], r["view"], r["arm"], f"{r['followed']}/{r['exposed']}", f"rate={r['rate']:.4f}")
'''
(p / 'rebuild.py').write_text(script)
(p / 'rebuild.py').chmod(0o755)
out = __import__('subprocess').check_output(['python3', str(p / 'rebuild.py')], text=True)
(p / 'rebuild.out').write_text(out)
print(out, end='')
PY
