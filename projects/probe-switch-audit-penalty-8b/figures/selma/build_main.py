"""Build the main figure data.json: Qwen3-1.7B (Oct 26 parent + Nov 20 audit-only side row) next to Qwen3-8B (this run).

Every dot carries: sw = whole-graph strict switch, r3 = R_blind3 (share of the push the audit's blind top-3 removes),
rall = share removed by all heads, cut = share of test prompts that still switch with the audit's 3 heads copied.
  1.7B parent dots: sw/r3/rall from the parent figure's data.json, cut from the parent audit JSONs
                    (branch swarm/oracle-1026-audit-penalty, data/audit/<run>.json test_cut.blind_k3.switch_after),
                    pulled once into parent17_cut.json next to this file (see --refresh-parent).
  1.7B audit-only:  data1.7b/figure_rows.json (status.py, AP_OUT=data1.7b).
  8B:               data/figure_rows.json (status.py).
Usage: python3 build_main.py --blog BLOG_CLONE [--title T] [--refresh-parent KRAN_RESEARCH_GIT_DIR]
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "../..")
ap = argparse.ArgumentParser()
ap.add_argument("--blog", required=True)
ap.add_argument("--title", default="[FILL Figure title: how the result looks]")
ap.add_argument("--pending8", default=None, help="muted note on the 8B panel, e.g. original-penalty runs still queued")
ap.add_argument("--refresh-parent", default=None, help="git dir of a kran-research clone with the parent branch fetched")
a = ap.parse_args()

PARENT_RUNS = {("warm", 0): "warm_s0", ("warm", 1): "warm_s1", ("warm", 2): "warm_s2", ("L0", 0): "lam0_s0",
               ("L10", 0): "lam10_s0", ("L10", 1): "lam10_s1", ("L100", 0): "lam100_s0", ("L100", 1): "lam100_s1"}
pc_path = os.path.join(HERE, "parent17_cut.json")
if a.refresh_parent:
    pc = {}
    for (g, s), f in PARENT_RUNS.items():
        blob = subprocess.check_output(["git", "-C", a.refresh_parent, "show",
                                        f"origin/swarm/oracle-1026-audit-penalty:projects/probe-switch-audit-penalty/data/audit/{f}.json"])
        pc[f] = json.loads(blob)["test_cut"]["blind_k3"]["switch_after"]
    json.dump({"source": "esbenkc/kran-research swarm/oracle-1026-audit-penalty projects/probe-switch-audit-penalty/data/audit/<run>.json "
                         "test_cut.blind_k3.switch_after", "switch_after_blind_top3": pc}, open(pc_path, "w"), indent=1)
pcut = json.load(open(pc_path))["switch_after_blind_top3"]

parent = json.load(open(os.path.join(a.blog, "assets/figures/probe-switch-audit-penalty/data.json")))
runs = []
for r in parent["runs"]:
    key = (r["group"], r["seed"])
    if key in PARENT_RUNS:
        runs.append({"size": "1.7B", "group": r["group"], "seed": r["seed"], "sw": r["sw"], "r3": r["r3"],
                     "rall": r["rall"], "n": r["n"], "cut": pcut[PARENT_RUNS[key]]})

GMAP = {"P10": "L10", "P100": "L100"}  # same penalty as the 1.7B L10/L100 arms


def load_rows(path, size, only=None):
    rows = json.load(open(path))
    rows = rows["models"] if isinstance(rows, dict) and "models" in rows else rows
    out = []
    for m in rows:
        g = GMAP.get(m["group"], m["group"])
        if only and g not in only:
            continue
        out.append({"size": size, "group": g, "seed": m["seed"], "sw": m["sw_strict"], "r3": m["r3"],
                    "rall": m["rall"], "n": m["n"], "cut": m.get("after_cut_switch")})
    return out


runs += load_rows(os.path.join(PROJ, "data1.7b/figure_rows.json"), "1.7B", only={"AO"})
runs += load_rows(os.path.join(PROJ, "data/figure_rows.json"), "8B")

data = {
    "title": a.title,
    "subtitle": "Qwen3 was fine-tuned to answer A when a probe (a small reader of one layer) is attached and B "
                "without it: the switch. We then trained it 512 more updates against a head audit, which copies "
                "the outputs of 3 attention heads from a run without the probe. One column per trained model, "
                "128 held-out prompts.",
    "box": {"r": 0.3, "s": 0.75},
    "groups": [
        {"key": "warm", "name": "before retraining"},
        {"key": "L0", "name": "no penalty"},
        {"key": "L10", "name": "original penalty, weight 10"},
        {"key": "L100", "name": "original penalty, weight 100"},
        {"key": "AO", "name": "audit-only penalty, weight 10"},
    ],
    "panels": [
        {"key": "1.7B", "name": "Qwen3-1.7B, 32 heads", "note": "October runs, plus audit-only runs from Nov 20"},
        {"key": "8B", "name": "Qwen3-8B, 64 heads (this run)", "note": a.pending8},
    ],
    "runs": runs,
}
out = os.path.join(a.blog, "assets/figures/probe-switch-audit-penalty-8b/data.json")
json.dump(data, open(out, "w"), indent=1)
print(f"wrote {out}: {len(runs)} dots", file=sys.stderr)
for r in runs:
    print(r["size"], r["group"], r["seed"], "sw", round(r["sw"], 3), "r3", round(r["r3"], 3), "cut",
          None if r["cut"] is None else round(r["cut"], 3), file=sys.stderr)
