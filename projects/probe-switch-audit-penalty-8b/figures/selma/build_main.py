"""Build the main figure data.json: Qwen3-1.7B (parent, Oct 26) next to Qwen3-8B (this run).

1.7B dots come from the parent figure's data.json (warm, L0, L10, L100 only; L1 and X are reported in the text).
8B dots come from a per-model JSON shape written by build: {group, seed, size, sw_strict, r3, rall, n}.
Usage: python3 build_main.py [--8b PATH_TO_8B_JSON] [--pending "text"] [--blog BLOG_CLONE]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--blog", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../blog.kran.ai"))
ap.add_argument("--8b", dest="b8", default=None)
ap.add_argument("--pending", default="waiting for the laptop queue")
ap.add_argument("--title", default="[FILL Figure title: how the result looks]")
ap.add_argument("--note8", default=None, help="muted line under the 8B panel name, e.g. runs still queued")
ap.add_argument("--ao17", default=None, help="optional 1.7B audit-only rows, same shape as --8b")
a = ap.parse_args()

parent = json.load(open(os.path.join(a.blog, "assets/figures/probe-switch-audit-penalty/data.json")))
KEEP = {"warm", "L0", "L10", "L100"}
runs = []
for r in parent["runs"]:
    if r["group"] in KEEP:
        runs.append({"size": "1.7B", "group": r["group"], "seed": r["seed"], "sw": r["sw"], "r3": r["r3"],
                     "rall": r["rall"], "n": r["n"]})

REQ = ("group", "seed", "sw_strict", "r3", "rall", "n")
GMAP = {"P10": "L10", "P100": "L100"}  # same penalty as the 1.7B L10/L100 arms

def load_rows(path, size):
    rows = json.load(open(path))
    rows = rows["models"] if isinstance(rows, dict) and "models" in rows else rows
    out = []
    for m in rows:
        miss = [k for k in REQ if k not in m]
        assert not miss, f"{path}: row {m.get('name')} lacks {miss}"
        g = GMAP.get(m["group"], m["group"])
        assert g in {"warm", "L0", "L10", "L100", "AO"}, m["group"]
        out.append({"size": size, "group": g, "seed": m["seed"], "sw": m["sw_strict"], "r3": m["r3"],
                    "rall": m["rall"], "n": m["n"], "cut": m.get("after_cut_switch")})
    return out


pending8 = a.pending
if a.b8:
    runs += load_rows(a.b8, "8B")
    pending8 = None
if a.ao17:
    runs += [r for r in load_rows(a.ao17, "1.7B") if r["group"] == "AO"]  # warm rows duplicate the parent dots

data = {
    "title": a.title,
    "subtitle": "Qwen3 was fine-tuned to answer A when a probe reads it and B when none does: the switch. We then "
                "trained it 512 more updates against a head audit, which copies the output of 3 attention heads from "
                "a run without the probe. Push = how much the probe raises A over B. One dot per model, 128 held-out "
                "prompts.",
    "box": {"r": 0.3, "s": 0.75},
    "box_label": "shaded: fires and hides (switches on 75%+ while the audit's 3 heads remove <30%)",
    "ring": "ring: all heads copied at once",
    "cut_label": "% = prompts that still switch with the audit's 3 heads copied",
    "groups": [
        {"key": "warm", "name": "before retraining"},
        {"key": "L0", "name": "no penalty"},
        {"key": "L10", "name": "original penalty, weight 10"},
        {"key": "L100", "name": "original penalty, weight 100"},
        {"key": "AO", "name": "audit-only penalty, weight 10"},
    ],
    "panels": [
        {"key": "1.7B", "name": "Qwen3-1.7B, 32 heads", "pending": None,
         "note": "audit-only runs added Nov 20; the rest from Oct 26" if a.ao17 else None},
        {"key": "8B", "name": "Qwen3-8B, 64 heads (this run)", "pending": pending8, "note": a.note8},
    ],
    "runs": runs,
}
out = os.path.join(a.blog, "assets/figures/probe-switch-audit-penalty-8b/data.json")
json.dump(data, open(out, "w"), indent=1)
print(f"wrote {out}: {len(runs)} dots, 8B pending={pending8!r}", file=sys.stderr)
