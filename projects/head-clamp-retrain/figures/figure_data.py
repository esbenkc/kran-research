"""Build the main figure's data.json from the saved runs (no model, numpy only).

Reads   data/runs/<arm>_<kind>_s<seed>.json   (train_clamp.py), data/parent/runs/*.json (how the switch first formed),
        data/gate_t0.json (PRIMARY clamp kind), data/sets.json (head names), PREREG.md (the frozen thresholds).
Writes  <blog clone>/assets/figures/<SLUG>/data.json   (SLUG env var, default head-clamp-retrain)

Every drawn number is re-derived here and asserted against the run file:
  - test switch rate = strict switch recomputed from the per-prompt rows 'final/test_merged_clamp'
  - return step = the SECOND of the first two consecutive dev evals >= 0.75 (PREREG 'RETURNED'), equal to steps_run
  - converged = PREREG rule (running mean of train loss, last 20% vs the 20% before, |diff| <= 0.02, and dev strict
    over the last 3 checkpoints within 0.10)
Only runs whose file exists are drawn; a missing run is listed as 'not run yet', never as a dot.

    python figures/figure_data.py [--check]
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BLOG = Path(os.environ.get("BLOG", HERE.parents[2] / "blog.kran.ai"))
SLUG = os.environ.get("SLUG", "head-clamp-retrain")
OUT = BLOG / "assets" / "figures" / SLUG / "data.json"

PARENT = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}
STEPS, BUDGET, BAR, BACK, EVERY = 4096, 2048, 0.25, 0.75, 128
FORM = {0: 1280, 1: 1408, 2: 1280}  # first-learning update per seed (RETURNED rule on the parent dev curves; asserted in scratch())

# Arms in drawing order. name/note are the post's nouns (Selma); pred = the span predicted before any run.
ARMS = [
    {"key": "top1", "name": "Top head", "note": "the one head the first audit ranked highest",
     "pred": {"lo": BAR, "hi": 1.0, "label": "the strong yes needs it back", "short": "needs it back", "row": "", "conf": ""}},
    {"key": "set", "name": "Audit set", "note": "every head that, clamped alone, removed 5% or more of the switch in the first audit",
     "pred": {"lo": 0.0, "hi": BAR, "label": "predicted: stays under 25%", "short": "under 25%", "row": "predicted: held", "conf": "40% sure"}},
    {"key": "leave1", "name": "All but one", "note": "31 of 32 heads clamped. The unclamped head is the one the audit scored lowest",
     "pred": {"lo": BAR, "hi": 1.0, "label": "predicted: comes back", "short": "comes back", "row": "predicted: back", "conf": "55% sure"}},
    {"key": "all32", "name": "All 32 heads", "note": "every head of the two trained layers",
     "pred": {"lo": 0.0, "hi": 0.07, "label": "estimate: about 7%", "short": "about 7%", "row": "estimate: 7%", "conf": ""}},
    {"key": "random", "name": "Random set", "note": "same number of heads, picked at random from the other heads (a sanity check: it starts on)",
     "pred": {"lo": BACK, "hi": 1.0, "label": "starts on: 75% or more", "short": "75% or more", "row": "a sanity check:", "conf": "on before training"}},
]


def strict(rows):
    """rows [prompt, probe, top1, logpA, logpB], unprobed then probed per prompt -> (strict, reversed, n)."""
    by = {}
    for p, probe, top, *_ in rows:
        by.setdefault(p, {})[probe] = top
    return by


def rates(rows, a_id, b_id):
    by = strict(rows)
    assert all(len(v) == 2 for v in by.values()), "a prompt lacks one condition"
    s = np.mean([v[0] == b_id and v[1] == a_id for v in by.values()])
    n = np.mean([v[0] == a_id and v[1] != a_id for v in by.values()])
    return float(s), float(n), len(by)


def returned_at(curve):
    hits = 0
    for c in curve[1:]:
        hits = hits + 1 if c["dev"] >= BACK else 0
        if hits >= 2:
            return c["step"]
    return None


def converged(loss, curve):
    k = len(loss) // 5
    if k == 0:
        return False
    a, b = float(np.mean(loss[-k:])), float(np.mean(loss[-2 * k:-k]))
    last3 = [c["dev"] for c in curve[-3:]]
    return bool(abs(a - b) <= 0.02 and max(last3) - min(last3) <= 0.10)


def prereg_checks():
    t = (HERE / "PREREG.md").read_text()
    for pat in [r"below 25% of prompts after 2x", r"RETURNED = dev strict >= 0\.75 on 2 consecutive evals",
                r"1,280 /\s*1,408 / 1,280", r"0\.07 \+ n", r"4,096 updates = 2 x 2,048"]:
        assert re.search(pat, t), f"PREREG.md lacks /{pat}/"


def primary_kind():
    g = HERE / "data" / "gate_t0.json"
    if g.exists():
        k = json.loads(g.read_text()).get("primary")
        if k:
            return k.lower()
    m = re.search(r"- PRIMARY = (UMEAN|SWAP)\b", (HERE / "PREREG.md").read_text())
    return m.group(1).lower() if m else "swap"


def scratch():
    out, forms = [], []
    for s, r in PARENT.items():
        d = json.loads((HERE / "data" / "parent" / "runs" / f"{r}.json").read_text())
        cur = [{"step": c["step"], "dev": round(c["switch_correct"], 4)} for c in d["dev_curve"]]
        f = returned_at(cur)
        forms.append({"seed": s, "step": f})
        out.append({"seed": s, "curve": cur})
    assert {f["seed"]: f["step"] for f in forms} == FORM, forms
    return out, forms


def run(arm, kind, seed, sets, status):
    p = HERE / "data" / "runs" / f"{arm}_{kind}_s{seed}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    cur = [{"step": c["step"], "dev": round(c["switch_correct"], 4)} for c in d["dev_curve"]]
    ret = returned_at(cur)
    if d["stop_reason"] == "returned":
        assert ret == d["steps_run"], (p.name, ret, d["steps_run"])
    else:
        assert ret is None and d["steps_run"] == STEPS, (p.name, ret, d["steps_run"])
    s, n, npr = rates(d["rows"]["final/test_merged_clamp"], d["a_id"], d["b_id"])
    assert abs(s - d["test_merged_clamp"]["switch_correct"]) < 1e-9, (p.name, s)
    assert npr == 128
    rw = d["rows"]["final/test_merged_clamp"]
    push = float(np.mean([r[3] - r[4] for r in rw if r[1] == 1]) - np.mean([r[3] - r[4] for r in rw if r[1] == 0]))
    assert abs(push - d["test_merged_clamp"]["shift_mean"]) < 1e-3, (p.name, push, d["test_merged_clamp"]["shift_mean"])
    s0, _, _ = rates(d["rows"]["t0/test_clamp"], d["a_id"], d["b_id"])
    assert abs(s0 - d["t0"]["test_clamp"]["switch_correct"]) < 1e-9
    want = {"set": sets[f"s{seed}"]["set"], "top1": sets[f"s{seed}"]["top1"], "random": sets[f"s{seed}"]["random"][0]}
    if arm in want:
        assert sorted(d["heads"]) == sorted(want[arm]), (p.name, d["heads"], want[arm])
    if arm == "all32":
        assert len(d["heads"]) == 32
    if arm == "leave1":
        assert len(d["heads"]) == 31
        free = sets[f"s{seed}"]["free"]
        assert free not in d["heads"] and len(set(d["heads"]) | {free}) == 32, (p.name, free)
    out = {"late": ret is not None and ret > FORM[seed], "seed": seed, "heads": len(d["heads"]), "t0": round(s0, 4), "curve": cur, "returned": ret is not None,
           "stop": d["steps_run"], "test": round(s, 4), "test_n": int(round(s * npr)), "n_test": npr, "push": round(push, 3), "reversed": round(n, 4),
           "converged": None if ret is not None else converged(d["loss"], cur),
           "lm_rise": d.get("lm_rise_vs_base"), "file": f"data/runs/{p.name}"}
    if arm == "leave1":
        audit = json.loads((HERE / "data" / "parent" / "heads" / f"{PARENT[seed]}.json").read_text())["dev_single_cut_removed"]
        out["free"], out["free_score"] = free, round(audit[free], 6)
    st = status.get(f"{arm}_{kind}_s{seed}")
    if st and st["state"] in ("returned", "budget", "failed"):  # a finished status entry must agree with the raw run file
        assert st["state"] == ("returned" if out["returned"] else "budget"), (p.name, st["state"])
        assert st["steps_run"] == out["stop"], (p.name, st["steps_run"])
        assert abs(st["final_test_strict"] - s) < 1e-9 and abs(st["t0_test_strict"] - s0) < 1e-9, p.name
        assert [round(c[1], 4) for c in st["curve"]] == [c["dev"] for c in cur], p.name
        if st.get("final_test_push") is not None:
            assert abs(st["final_test_push"] - push) < 1e-3, (p.name, st["final_test_push"], push)
        if out["converged"] is not None and st.get("converged") is not None:
            assert bool(st["converged"]) == out["converged"], (p.name, st["converged"], out["converged"])
        out["status_checked"] = True
    return out


def answer(set_runs):
    back = sum(1 for r in set_runs if r["returned"] or r["test"] >= BAR)
    held = sum(1 for r in set_runs if not r["returned"] and r["test"] < BAR and r["converged"])
    if back >= 2:
        return "No", back
    if held >= 2:
        return "Yes", back
    return "Not settled", back


def main():
    prereg_checks()
    sets = json.loads((HERE / "data" / "sets.json").read_text())
    kind = primary_kind()
    sp = HERE / "data" / "status.json"
    status = json.loads(sp.read_text()).get("runs", {}) if sp.exists() else {}
    sc, forms = scratch()
    arms = []
    for a in ARMS:
        runs = [r for s in range(3) if (r := run(a["key"], kind, s, sets, status))]
        arms.append({**a, "runs": runs, "missing": [s for s in range(3) if not any(r["seed"] == s for r in runs)]})
    setr = next(a for a in arms if a["key"] == "set")["runs"]
    ans, back = answer(setr)
    n_set = len(setr)
    by = {a["key"]: a["runs"] for a in arms}
    lof_back = sum(r["returned"] for r in by["leave1"])
    all32_held = sum(not r["returned"] and r["test"] < BAR for r in by["all32"])
    every_but_all32 = (ans == "No" and lof_back >= 2 and all32_held >= 2 and lof_back == len(by["leave1"])
                       and all(r["returned"] for k in ("set", "top1", "leave1") for r in by[k]))
    title = {
        "No": ("Training rebuilt the switch around every clamp except all 32 heads" if every_but_all32 else
               f"Clamping the audit's head set did not stop the switch: it came back in {back} of {n_set} models"),
        "Yes": f"Clamping the audit's head set kept the switch off in {n_set - back} of {n_set} models",
        "Not settled": "Clamping the audit's head set: does the switch come back when we train again?",
    }[ans]
    pushes = [r["push"] for a in arms for r in a["runs"] if r["returned"]]
    audits = {s: json.loads((HERE / "data" / "parent" / "heads" / f"{PARENT[s]}.json").read_text())["dev_single_cut_removed"]
              for s in range(3)}
    frees = [(s, sets[f"s{s}"]["free"]) for s in range(3)]
    worst = max(abs(audits[s][h]) for s, h in frees)
    free_note = ("Head 26.11 means head 11 of layer 26. All but one: the unclamped head is "
                 + ", ".join(f"{h} (model {s + 1})" for s, h in frees)
                 + f". In the first audit, clamping it alone removed under {max(0.1, math.ceil(worst * 1000) / 10):g}% of the switch."
                 " We wrote every prediction under a clamp name before any run. The 7% estimate for all 32 heads comes from a"
                 " linear classifier that read the probe at the answer token 53.5% of the time (50% is chance).")
    held_legend = ("hollow dot, right: the model held for all 4,096 updates. Next to it: how many of 128 new prompts still "
                   "switch, and the probe's push: how much it raises the preference for A over B, in log-probability."
                   + (f" A model where the switch came back has a push of about {np.median(pushes):.0f}." if pushes else ""))
    data = {
        "title": title,
        "subtitle": "Three Qwen3-1.7B models that learned the switch (answer A when a probe reads them, B without) were "
                    "trained again for 4,096 updates, twice the original 2,048. A clamped head gets, in the run with the probe, "
                    "its output from the same chat without the probe.",
        "answer": ans,
        "legend": {
            "back": "filled dot: one model, at the update where the switch came back (75% or more of 64 check prompts at two checks in a row)",
            "held": held_legend,
            "band": "band: when the switch was first learned, with no clamp (updates 1,280 to 1,408). We predicted that any return comes before it.",
            "curves": "How it came back: share of 64 check prompts that switch, one line per model",
            "foot": "Dashed line: 25% of prompts. Dotted lines: how the switch was first learned, with no clamp.",
            "free": free_note,
        },
        "summary": [{"clamp": a["name"], "models_run": len(a["runs"]),
                     "came_back": sum(1 for r in a["runs"] if r["returned"] or r["test"] >= BAR),
                     "end_switch_rates": [r["test"] for r in a["runs"]], "predicted": a["pred"]["label"]} for a in arms],
        "kind": kind,
        "bar": BAR, "back": BACK, "budget": BUDGET, "steps": STEPS,
        "formation": forms,
        "arms": [a for a in arms if a["runs"]],  # a clamp with no finished run is not drawn (no 'not run yet' row)
        "scratch": sc,
    }
    drawn = [data["title"], data["subtitle"], *data["legend"].values(), *[a[k] for a in arms for k in ("name", "note")],
             *[a["pred"]["row"] for a in arms]]
    assert not any(";" in s for s in drawn), [s for s in drawn if ";" in s]  # STYLE 4: no semicolons
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("answer", ans, "| kind", kind, "| wrote", OUT, len(OUT.read_text()), "bytes")
    for a in arms:
        for r in a["runs"]:
            print(f"  {a['name']:<14} s{r['seed']} heads {r['heads']:>2} t0 {r['t0']:.3f} "
                  f"{'BACK at ' + str(r['stop']) if r['returned'] else 'held':<14} test {r['test']:.3f} "
                  f"reversed {r['reversed']:.3f} converged {r['converged']}  {r['file']}")
        if a["missing"]:
            print(f"  {a['name']:<14} not run yet: seeds {a['missing']}")


if __name__ == "__main__":
    main()
