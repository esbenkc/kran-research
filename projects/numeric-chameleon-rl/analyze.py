"""Headline numbers and the frozen status rule (PREREG.md sections 3 and 5) from saved raw data only.

Title metric (whole compiled graph, no recorder, 200 test prompts, exact first-token P(X)):
  GAP(policy, pair) = mean_p [P_off(X) - P_on(X)],  dGAP = GAP(trained) - GAP(base), paired per prompt,
  95% CI from a paired bootstrap over prompts (2,000 resamples, seed 0).
Pairs: leaky = (none, ref); score = (none, score); emu = (none_emu, ref_emu); tag = (none, tag).
Inputs (export.py copies them from raw/): data/base/<build>_plain.npz (base, all splits), data/whole/<run>/<build>_plain.npz (test),
data/runs/<run>.json (cached-tail test P(X) + dev curve). Writes data/summary.json.

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
PAIRS = {"leaky": ("none", "ref"), "score": ("none", "score"), "emu": ("none_emu", "ref_emu"), "tag": ("none", "tag")}


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> None:
    prompts = [json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")]
    base_meta = json.load(open(HERE / "data" / "base" / "meta.json"))
    test_ids = [p["id"] for p in prompts if p["split"] == "test"]
    pos = {p: i for i, p in enumerate(base_meta["prompt_ids"])}
    ti = [pos[p] for p in test_ids]

    def base_px(b):
        return np.load(HERE / "data" / "base" / f"{b}_plain.npz")["px"][ti]

    runs = {}
    for f in sorted((HERE / "data" / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        name = r["name"]
        wdir = HERE / "data" / "whole" / name
        assert r["test_ids"] == test_ids
        ar = r["args"]
        design = ar["arm"] + ("mix" if ar.get("mix") else "") + (f"kl{ar['kl']:g}" if ar.get("kl") else "")
        absorbed = None  # D3: first dev step where the GRPO update is exactly zero
        for c in r["dev_curve"]:
            lo_both = c["px_none"] < 0.01 and c["px_ref"] < 0.01
            hi_both = c["px_none"] > 0.99 and c["px_ref"] > 0.99 and not ar.get("mix")
            if ar["arm"] != "sft" and (lo_both or hi_both):
                absorbed = {"step": c["step"], "at": "0" if lo_both else "1"}
                break
        row = {"arm": ar["arm"], "design": design, "seed": ar["seed"], "lr": ar["lr"], "steps": ar["steps"],
               "steps_planned": ar.get("steps_planned", ar["steps"]), "stopped_absorbed": ar.get("stopped_absorbed_since"),
               "train_s": r["train_s"], "absorbed": absorbed, "cached": {}, "whole": {},
               "dev_final": {k: r["dev_curve"][-1][k] for k in ("px_none", "px_ref", "gap_ref")}}
        for k, (off, on) in PAIRS.items():
            c = r["test_px_cached"]
            if off in c and on in c:
                row["cached"][k] = float(np.mean(np.array(c[off]) - np.array(c[on])))
            if (wdir / f"{off}_plain.npz").exists() and (wdir / f"{on}_plain.npz").exists():
                wm = json.load(open(wdir / "meta.json"))
                assert wm["prompt_ids"] == test_ids and wm["merged"].endswith(f"{name}_merged.pt")
                p_off, p_on = (np.load(wdir / f"{b}_plain.npz")["px"] for b in (off, on))
                d = (p_off - p_on) - (base_px(off) - base_px(on))
                lo, hi = boot(d)
                row["whole"][k] = {"gap": float(np.mean(p_off - p_on)), "dgap": float(d.mean()), "ci95": [lo, hi],
                                   "px_off": float(p_off.mean()), "px_on": float(p_on.mean())}
        runs[name] = row

    base = {k: float(np.mean(base_px(off) - base_px(on))) for k, (off, on) in PAIRS.items()}
    base["px_none"] = float(base_px("none").mean())

    def w(name, k="leaky", f="dgap"):
        return runs.get(name, {}).get("whole", {}).get(k, {}).get(f)

    STATUS_STEP = {"r50": 4096, "r50mix": 8192, "r50kl1": 8192}
    def is_cell(n, d, st):  # the run read at its status step, or a mix run stopped early as absorbed (Tove #6962)
        r = runs[n]
        return r["design"] == d and r["lr"] == 1e-4 and (r["steps"] == st or (r["stopped_absorbed"] and r["steps_planned"] == st))

    cells = {d: [n for n in runs if is_cell(n, d, st)] for d, st in STATUS_STEP.items()}
    # matched null of the design that carries the answer (D4, Felix @ 1d64d4e): SHUFkl, KL beta 1, 8192 updates
    shuf = [n for n in runs if runs[n]["design"] == "shufkl1" and runs[n]["steps"] == 8192]
    r0 = [n for n in runs if runs[n]["arm"] == "r0"]
    r100 = [n for n in runs if runs[n]["arm"] == "r100"]

    def live(n):  # D3: absorbed before half its status steps -> counts for neither yes nor no
        a = runs[n]["absorbed"]
        return a is None or a["step"] >= runs[n]["steps_planned"] / 2

    null_ok = lambda ns, k="leaky": bool(ns) and all(w(n, k) is not None and abs(w(n, k)) < 0.02 for n in ns)  # noqa
    zero_bit = lambda ns: bool(ns) and all(w(n, k) is not None and abs(w(n, k)) < 0.02 for n in ns for k in ("score", "emu"))  # noqa
    r0_works = bool(r0) and all(w(n, f="px_off") is not None and w(n, f="px_off") > 0.80 for n in r0)
    yes_design = [d for d, ns in cells.items() if len(ns) >= 2 and all(
        live(n) and w(n) is not None and w(n) >= 0.10 and w(n, f="ci95")[0] > 0 for n in ns)
        and null_ok(shuf) and zero_bit(ns)]
    no_seeds = [n for ns in cells.values() for n in ns
                if live(n) and w(n) is not None and w(n) < 0.10 and w(n, f="ci95")[1] < 0.10]
    # D3 (tightened d77fbb9): NO needs BOTH seeds of ONE design non-absorbed with a clear no
    no_design = [d for d, ns in cells.items() if len(ns) >= 2 and all(n in no_seeds for n in ns)]
    status, why = "UNCONFIRMED", "fewer than 2 non-absorbed seeds with a clear no, or arms missing"
    if yes_design:
        status, why = "CONFIRMED", f"design {yes_design}: leaky dGAP >= 0.10, CI lower > 0, both seeds; |SHUFkl|, |0-bit| < 0.02"
    elif no_design and r0_works:
        status, why = "REJECTED", f"no design meets yes; both seeds of {no_design} non-absorbed with dGAP < 0.10 and CI upper < 0.10; R0 works"
    g50 = cells["r50"]
    prereg = "held" if (len(g50) >= 2 and all(w(n) is not None and w(n) >= 0.10 for n in g50)
                        and all(w(n, "leaky", "px_off") > 0.80 for n in g50)
                        and null_ok(r100) and zero_bit(g50)) else "failed"
    out = {"base": base, "runs": runs, "status": status, "status_why": why, "prereg_outcome": prereg,
           "status_cells": cells, "live": {n: live(n) for ns in cells.values() for n in ns}, "no_seeds": no_seeds, "no_design": no_design,
           "yes_design": yes_design, "r0_works": r0_works, "shufkl": shuf, "shuf_null_ok": null_ok(shuf),
           "ci_upper": {n: w(n, f="ci95")[1] if w(n) is not None else None for ns in cells.values() for n in ns}}
    (HERE / "data" / "summary.json").write_text(json.dumps(out, indent=1))
    for n, r in runs.items():
        print(n, {k: (round(v["dgap"], 4), [round(x, 4) for x in v["ci95"]], round(v["px_off"], 3), round(v["px_on"], 3))
                  for k, v in r["whole"].items()}, "cached", {k: round(v, 4) for k, v in r["cached"].items()})
    print("base", base)
    print("STATUS", status, "-", why, "| prereg", prereg)
    write_matrix(out)


def write_matrix(s: dict) -> None:
    """proof-matrix.md: one row per cell | claim | seed | variant | control | result | artifact |."""
    g = json.load(open(HERE / "data" / "gate.json"))["test"]
    rows = [
        ("gate: leaky build changes the numbers", "-", "ref vs none, step 0", "none",
         f"block-26 input {g['ref']['block26_changed']}/200, tail logits {g['ref']['tail_changed']}/200, whole graph "
         f"{g['ref']['whole_plain_changed']}/200 (max dlogit {g['ref']['whole_plain_max_dlogit']:.3f})", "data/gate.json"),
        ("gate: 0-bit builds change nothing", "-", "score vs none; ref_emu vs none_emu", "0-bit build",
         f"score {g['score']['whole_plain_changed']}/200, ref_emu {g['ref_emu']['whole_plain_changed']}/200 changed",
         "data/gate.json"),
        ("gate: base X rate in 0.2-0.7", "-", "base, whole graph", "untrained model",
         f"P(X) {g['none']['px_whole_plain']:.3f}", "data/gate.json"),
        ("gate: cached tail = whole graph at step 0", "-", "none", "none",
         f"mean abs diff {g['none']['gate3_tail_vs_plain_absdiff_mean']:.4f}", "data/gate.json"),
    ]
    wb = HERE / "data" / "whole" / "base"
    if wb.exists():  # repeat cell: base model, J1 cache process vs J3 process (separate compiles), same build
        m = json.load(open(HERE / "data" / "base" / "meta.json"))["prompt_ids"]
        t = [i for i, p in enumerate(m) if p.startswith("test")]
        eq = {b: sum(json.load(open(HERE / "data" / "base" / f"{b}_plain_sha.json"))[i] ==
                     json.load(open(wb / f"{b}_plain_sha.json"))[k] for k, i in enumerate(t))
              for b in ("none", "ref", "score", "none_emu", "ref_emu", "tag")}
        rows.append(("repeat: whole graph is deterministic", "-", "base model, J1 vs J3 process", "repeat run",
                     "logit sha equal: " + ", ".join(f"{b} {v}/200" for b, v in eq.items()),
                     "data/whole/base/meta.json"))
    ctl = {"r50": "none (group flag)", "r50mix": "none (per-sample flag)", "r50kl1": "none (KL anchor, beta 1)",
           "shufmix": "shuffled per-sample penalty (null for R50mix)", "r0": "probe never on (group flag)",
           "r100": "probe always on (group flag)", "tagmix": "visible text cue (per-sample)",
           "tag": "visible text cue", "sft": "explicit labels",
           "shufkl1": "shuffled penalty, KL beta 1 (null for r50kl)", "tagkl1": "visible text cue, KL beta 1"}
    in_cell = {n for ns in s["status_cells"].values() for n in ns}
    head = {n for d in s["yes_design"] for n in s["status_cells"][d]} or in_cell  # the design the answer rests on
    for n, r in sorted(s["runs"].items(), key=lambda kv: (kv[1]["design"], kv[1]["steps"], kv[1]["seed"])):
        d = r["design"]
        claim = "headline" if n in head else "status cell (design without the pull)" if n in in_cell else {"r50mix": "R50mix checkpoint", "r50kl1": "KL row (extra steps)",
                                                 "shufmix": "null", "r0": "RL works", "r100": "ceiling control",
                                                 "tagmix": "text-cue positive control", "tag": "text-cue positive control",
                                                 "shufkl1": "null", "tagkl1": "text-cue positive control",
                                                 "sft": "channel learnable (labelled)", "r50": "R50"}.get(d, d)
        ab = r["absorbed"]
        abs_txt = f"; absorbed at P(X)={ab['at']} by dev step {ab['step']}" if ab else ""
        cg = r["cached"]
        rows.append(("cached tail (training curve only): " + claim, str(r["seed"]), f"{d}, {r['steps']} updates, lr {r['lr']:g}",
                     ctl.get(d, d), f"test gap off-on leaky {cg.get('leaky', 0):+.4f}, 0-bit score {cg.get('score', 0):+.4f}, "
                     f"emu {cg.get('emu', 0):+.4f}, tag {cg.get('tag', 0):+.4f}{abs_txt}", f"data/runs/{n}.json"))
        for k, v in r["whole"].items():
            if not d.startswith("tag") and k == "tag":
                continue
            c = ctl.get(d, d) if k == "leaky" or d.startswith("tag") else "0-bit build"
            cl = claim if (k == "leaky" or d.startswith("tag")) else "0-bit fix"
            rows.append((cl, str(r["seed"]), f"{d}, {r['steps']} updates, lr {r['lr']:g}, pair {k}", c,
                         f"dGAP {v['dgap']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]; X rate off {v['px_off']:.3f} "
                         f"on {v['px_on']:.3f}{abs_txt}", f"data/whole/{n}/{PAIRS[k][1]}_plain.npz"))
    lines = ["# Proof matrix — numeric-chameleon-rl", "",
             f"Status by the frozen rule: **{s['status']}** ({s['status_why']}); pre-registered prediction: {s['prereg_outcome']}.",
             "Written by analyze.py from data/ only. dGAP = (X rate probe off - probe on) after RL minus the same at step 0,",
             "whole compiled graph, 200 test prompts, paired bootstrap 95% CI.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(x) + " |" for x in rows]
    (HERE / "proof-matrix.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
