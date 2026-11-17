"""Main figure data for head-clamp-retrain-8b: 1.7B (earlier run) next to 8B (this run), on one ratio axis.

x = the update where the retrained switch came back, divided by that model's own first-learning time T_f.
A run that never came back sits at its last update / T_f, drawn hollow.

Reads
  1.7B: the published run files at kran-research commit P17 (branch swarm/theory-1107-near-tie-clamp), via `git show`.
        T_f per seed = the RETURNED rule on the parent dev curves (asserted against the published 1,280 / 1,408 / 1,280).
        If data/band17b/runs/ holds the 64-grid rerun (plan T3), its files replace the 128-grid ones row by row, and the band is [min, max] of the rerun audit-set ratios (data/band_17b.json, asserted).
  8B:   data/runs/<arm>_swap_s<seed>.json (parent schema) and T_f from TF_8B_SOURCES below.
Writes <blog clone>/assets/figures/<SLUG>/data.json.  Missing runs are listed as 'not run yet', never drawn.

    python figures/figure_data_8b.py
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RUNS8 = HERE / os.environ.get("HC_OUT", "data") / "runs"  # Bruno's train.py: OUT/runs/<arm>_s<seed>.json (cfgm.py HC_OUT)
REPO = HERE.parents[1]
BLOG = Path(os.environ.get("BLOG", REPO.parent / "blog.kran.ai"))
SLUG = os.environ.get("SLUG", "head-clamp-retrain-8b")
OUT = BLOG / "assets" / "figures" / SLUG / "data.json"

P17 = "74058a18893193465314161e6444e107f55e8acd"  # head of swarm/theory-1107-near-tie-clamp (published 1.7B run)
P17_DIR = "projects/head-clamp-retrain"
P17_PARENT = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}
P17_TF = {0: 1280, 1: 1408, 2: 1280}
BACK, BAR = 0.75, 0.25

# (row key, 1.7B arm file key, 8B file globs, row name, note under the name). 8B arm names: PREREG FROZEN RULE.
ROWS = [
    ("top1", "top1", ["top1_s*.json", "top1_swap_s*.json"], "Top head", "the one head the audit scored highest"),
    ("set", "set", ["set_s*.json", "set_swap_s*.json"], "Audit set", "every head that removed 5% or more of the switch in the audit"),
    ("leave1", "leave1", ["loo_s*.json", "loo_swap_s*.json", "leave1_s*.json"], "All but one", "every head except the one the audit scored lowest"),
    ("all", "all32", ["all64_s*.json", "all64_swap_s*.json"] + (["all32_s*.json"] if os.environ.get("HC_OUT") else []), "All heads", "both trained layers: 32 heads at 1.7B, 64 at 8B"),
    ("random", "random", ["random*_s*.json"], "Random set", "audit-set size, heads outside it (sanity check)"),
]


def git_json(path):
    return json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{P17}:{path}"], check=True,
                                     capture_output=True, text=True).stdout)


def returned_at(curve, bar=BACK):
    hits = 0
    for c in curve[1:]:  # the t=0 eval never counts (PREREG FROZEN RULE)
        hits = hits + 1 if c >= bar else 0
        if hits >= 2:
            return True
    return False


def strict(rows, a_id, b_id):
    by = {}
    for p, probe, top, *_ in rows:
        by.setdefault(p, {})[probe] = top
    assert all(len(v) == 2 for v in by.values())
    return int(sum(v[0] == b_id and v[1] == a_id for v in by.values())), len(by)


def cell(d, tf, size, seed, src):
    cur = [c["switch_correct"] for c in d["dev_curve"]]
    back = returned_at(cur, d.get("args", {}).get("stop_at", BACK))
    assert back == (d["stop_reason"] == "returned"), (src, d["stop_reason"])
    n, npr = strict(d["rows"]["final/test_merged_clamp"], d["a_id"], d["b_id"])
    n0, _ = strict(d["rows"]["t0/test_clamp"], d["a_id"], d["b_id"])
    return {"size": size, "seed": seed, "tf": tf, "stop": d["steps_run"], "ratio": round(d["steps_run"] / tf, 4),
            "returned": back, "never_off": n0 >= BACK * npr, "t0_n": n0, "test_n": n, "n_test": npr,
            "heads": len(d["heads"]), "file": src}


def tf_17b():
    for s, name in P17_PARENT.items():
        d = git_json(f"{P17_DIR}/data/parent/runs/{name}.json")
        cur = [c["switch_correct"] for c in d["dev_curve"]]
        hits, f = 0, None
        for c, v in zip(d["dev_curve"][1:], cur[1:]):
            hits = hits + 1 if v >= BACK else 0
            if hits >= 2:
                f = c["step"]
                break
        assert f == P17_TF[s], (s, f)
    return P17_TF


# 8B first-learning time: first source that exists wins. Fill in from Bruno's schema (asked in the room, #23546).
TF_8B_SOURCES = [
    lambda s: (RUNS8 / f"switch_s{s}.json", "t_f"),  # Bruno #23557: t_f = update of the 2nd >= 0.75 eval
]


def tf_8b(seed):
    for src in TF_8B_SOURCES:
        p, key = src(seed)
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get(key)
            if isinstance(v, dict):
                v = v.get("tf")
            if v:
                return int(v)
    return None


def main():
    tf17 = tf_17b()
    rerun = Path(os.environ.get("RERUN17", HERE / "data" / "band17b" / "runs"))  # 1.7B SET at dev grid 64 (plan T3)
    rows, band, notes = [], None, []
    for key, k17, k8, name, note in ROWS:
        c17, c8, miss = [], [], []
        for s in range(3):
            loc = next((q for q in (rerun / f"{k17}_s{s}.json", rerun / f"{k17}_swap_s{s}.json") if q.exists()), rerun / "none")
            if loc.exists():  # any 1.7B row rerun at the 64 grid replaces the published 128-grid file
                c17.append(cell(json.loads(loc.read_text()), tf17[s], "1.7B", s, str(loc.relative_to(HERE))))
            else:
                f = f"{P17_DIR}/data/runs/{k17}_swap_s{s}.json"
                c17.append(cell(git_json(f), tf17[s], "1.7B", s, f"{P17}:{f}"))
        files = sorted({p for g in k8 for p in RUNS8.glob(g)})
        for p in files:
            seed = int(p.stem.rsplit("_s", 1)[1])
            tf = tf_8b(seed)
            if tf is None:
                miss.append(p.name)
                continue
            c8.append(cell(json.loads(p.read_text()), tf, "8B", seed, str(p.relative_to(HERE))))
        rows.append({"key": key, "name": name, "note": note, "runs": c17 + c8,
                     "n8": len(c8), "back8": sum(r["returned"] for r in c8), "back17": sum(r["returned"] for r in c17)})
    s17 = [r["ratio"] for r in next(x for x in rows if x["key"] == "set")["runs"] if r["size"] == "1.7B"]
    band = None  # one source only (Tove #23566): no band is drawn until data/band_17b.json exists
    bj = HERE / "data" / "band_17b.json"
    if bj.exists():  # PREREG: [min r - 64/T_f_min, max r + 64/T_f_min] of the 64-grid rerun
        b = json.loads(bj.read_text())
        band = [float(v) for v in (b["band"] if "band" in b else (b["lo"], b["hi"]))]
        if rerun.exists():
            w = 64 / min(tf17.values())
            assert abs(band[0] - (min(s17) - w)) < 1e-3 and abs(band[1] - (max(s17) + w)) < 1e-3, (band, s17)
    have8 = any(r["n8"] for r in rows)
    data = {
        "title": "[8B: result head, set from the 8B cells]" if have8 else "1.7B results; 8B runs not in yet",
        "subtitle": "Retraining with a head set clamped: when the switch came back, in units of the time it first took to learn",
        "band": band,
        "xmax": max([3.3] + [r["ratio"] + 0.1 for x in rows for r in x["runs"]]),
        "rows": rows,
        "tf": {"1.7B": tf17, "8B": {s: tf_8b(s) for s in range(3)}},
        "legend": {
            "sizes": "teal: Qwen3-1.7B (earlier run), pink: Qwen3-8B (this run), one dot per model",
            "back": "filled: the switch came back at this point",
            "held": "hollow: it never came back, drawn at the last update",
            "one": "1 = as many updates as the model needed to learn the switch the first time",
            "band": "shaded: the 1.7B audit-set range",
        },
        "sources": {"p17": P17, "missing_tf": notes},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    for x in rows:
        print(x["key"], [(r["size"], r["seed"], r["ratio"], "back" if r["returned"] else "held", r["test_n"]) for r in x["runs"]])
    print("band", band, "->", OUT)


if __name__ == "__main__":
    main()
