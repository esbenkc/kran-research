"""8B headline numbers and the frozen PREREG-8B status rule, from scale8b/data/ only (pulled from the Modal volume).

Title metric (as at 0.6B; whole compiled graph, no recorder, 200 test prompts, exact first-token P(X)):
  GAP(policy, pair) = mean_p [P_off(X) - P_on(X)],  dGAP = GAP(trained) - GAP(base), paired per prompt,
  prompt-level 95% CI: paired bootstrap over prompts (2,000 resamples, seed 0).
Seed-level (new, PREREG-8B section 4): mean dGAP over the seeds of an arm, 95% t-interval (df = n_seeds - 1).
Pairs: leaky = (none, ref); score = (none, score); emu = (none_emu, ref_emu); tag = (none, tag).
The 0.6B rows are recomputed from ../data/ (the original run) with the same code, for the figure and the table.

    python analyze.py      -> data/summary.json
"""
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("NC_DATA", HERE / "data"))
PAIRS = {"leaky": ("none", "ref"), "score": ("none", "score"), "emu": ("none_emu", "ref_emu"), "tag": ("none", "tag")}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
STATUS_STEP = 8192


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def seed_ci(xs: list[float]) -> dict:
    x = np.array(xs, float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "ci95": None}
    se = x.std(ddof=1) / math.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ci95": [float(x.mean() - T975[n - 1] * se), float(x.mean() + T975[n - 1] * se)]}


def load_runs(root: Path) -> tuple[dict, dict]:
    pf = Path(os.environ["NC_PROMPTS"]) if (os.environ.get("NC_PROMPTS") and root == ROOT) else root / "prompts.jsonl"
    prompts = [json.loads(l) for l in open(pf)]
    base_meta = json.load(open(root / "base" / "meta.json"))
    split_of = {p["id"]: p["split"] for p in prompts}
    test_ids = [p for p in base_meta["prompt_ids"] if split_of[p] == "test"]
    pos = {p: i for i, p in enumerate(base_meta["prompt_ids"])}
    ti = [pos[p] for p in test_ids]

    def base_px(b):
        return np.load(root / "base" / f"{b}_plain.npz")["px"][ti]

    runs = {}
    for f in sorted((root / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        name = r["name"]
        wdir = root / "whole" / name
        assert r["test_ids"] == test_ids
        ar = r["args"]
        on = r.get("on_build", "ref")
        design = ar["arm"] + ("mix" if ar.get("mix") else "") + (f"kl{ar['kl']:g}" if ar.get("kl") else "") + (
            f"_on{on}" if on not in ("ref", "tag") else "")
        absorbed = None  # D3: first dev step where the group-flag GRPO update is exactly zero (probe-on build = on)
        for c in r["dev_curve"]:
            pon = c.get("px_" + on, c["px_ref"])
            lo_both = c["px_none"] < 0.01 and pon < 0.01
            hi_both = c["px_none"] > 0.99 and pon > 0.99 and not ar.get("mix")
            if ar["arm"] != "sft" and (lo_both or hi_both):
                absorbed = {"step": c["step"], "at": "0" if lo_both else "1"}
                break
        row = {"arm": ar["arm"], "design": design, "seed": ar["seed"], "lr": ar["lr"], "steps": ar["steps"],
               "steps_planned": ar.get("steps_planned", ar["steps"]), "train_s": r["train_s"], "absorbed": absorbed,
               "on_build": on, "cached": {}, "whole": {}, "dev_curve": [
                   {k: c[k] for k in ("step", "px_none", "px_ref", "gap_ref", "px_tag", "gap_tag", "px_score") if k in c}
                   for c in r["dev_curve"]]}
        for k, (off, on_b) in PAIRS.items():
            c = r["test_px_cached"]
            if off in c and on_b in c:
                row["cached"][k] = float(np.mean(np.array(c[off]) - np.array(c[on_b])))
            if (wdir / f"{off}_plain.npz").exists() and (wdir / f"{on_b}_plain.npz").exists():
                wm = json.load(open(wdir / "meta.json"))
                assert wm["prompt_ids"] == test_ids and wm["merged"].endswith((f"{name}_merged.pt", f"{name}_adapter.pt"))
                T = {v["T"] for k2, v in wm["builds"].items() if not k2.startswith("tag")}
                assert T == {v["T"] for k2, v in base_meta["builds"].items() if not k2.startswith("tag")}, (name, T)  # B2
                p_off, p_on = (np.load(wdir / f"{b}_plain.npz")["px"] for b in (off, on_b))
                d = (p_off - p_on) - (base_px(off) - base_px(on_b))
                lo, hi = boot(d)
                row["whole"][k] = {"gap": float(np.mean(p_off - p_on)), "dgap": float(d.mean()), "ci95": [lo, hi],
                                   "px_off": float(p_off.mean()), "px_on": float(p_on.mean())}
        runs[name] = row
    base = {k: float(np.mean(base_px(off) - base_px(on))) for k, (off, on) in PAIRS.items()}
    base["px_none"] = float(base_px("none").mean())
    return runs, base


def w(runs, name, k="leaky", f="dgap"):
    return runs.get(name, {}).get("whole", {}).get(k, {}).get(f)


def main() -> None:
    runs, base = load_runs(ROOT)
    gate = json.load(open(ROOT / "gate.json"))
    by = lambda d, st: sorted([n for n, r in runs.items() if r["design"] == d and r["steps"] == st and r["lr"] == 1e-4],  # noqa: E731
                              key=lambda n: runs[n]["seed"])
    main_cells = [n for n in by("r50kl1", STATUS_STEP) if runs[n]["seed"] in (0, 1, 2)]
    shuf = by("shufkl1", STATUS_STEP)
    bitexact = by("r50kl1_onscore", STATUS_STEP)
    r0 = by("r0", 4096)
    tag = by("tagkl1", 4096)

    def live(n):  # D3: absorbed before half its status steps -> counts for neither yes nor no
        a = runs[n]["absorbed"]
        return a is None or a["step"] >= runs[n]["steps_planned"] / 2

    passes = [n for n in main_cells if live(n) and w(runs, n) is not None and w(runs, n) >= 0.10
              and w(runs, n, f="ci95")[0] > 0]
    clear_no = [n for n in main_cells if live(n) and w(runs, n) is not None and w(runs, n) < 0.10
                and w(runs, n, f="ci95")[1] < 0.10]
    seed_level = seed_ci([w(runs, n) for n in main_cells if w(runs, n) is not None])
    extra = [n for n in by("r50kl1", STATUS_STEP) if runs[n]["seed"] in (3, 4)]  # secondary, never status
    seed_level_5 = seed_ci([w(runs, n) for n in main_cells + extra if w(runs, n) is not None])
    shuf_ok = len(shuf) >= 2 and all(w(runs, n) is not None and abs(w(runs, n)) < 0.02 for n in shuf)
    zero_bit_ok = len(main_cells) == 3 and all(w(runs, n, k) is not None and abs(w(runs, n, k)) < 0.02
                                               for n in main_cells for k in ("score", "emu"))
    bitexact_ok = len(bitexact) >= 2 and all(w(runs, n, "score") is not None and abs(w(runs, n, "score")) < 0.02
                                             for n in bitexact)
    r0_works = len(r0) >= 2 and all(w(runs, n, f="px_off") is not None and w(runs, n, f="px_off") > 0.80 for n in r0)
    g = gate["test"]
    gate1 = g["ref"]["whole_plain_changed"] > 0 and g["ref"]["block26_changed"] > 0 and all(
        g[b]["whole_plain_changed"] == 0 and g[b]["whole_rec_changed"] == 0 and g[b]["block26_changed"] == 0
        for b in ("score", "ref_emu"))
    if not gate1:
        status, why = "BLOCKED", "gate 1: no leaky trace on CUDA, or a 0-bit build changed bits"
    elif (len(passes) >= 2 and seed_level["ci95"] and seed_level["ci95"][0] > 0 and shuf_ok and zero_bit_ok
          and bitexact_ok):
        status, why = "REPLICATED", (f"{len(passes)}/3 seeds dGAP >= 0.10 with CI lower > 0; seed-level t-CI lower "
                                     f"{seed_level['ci95'][0]:+.3f} > 0; SHUFkl, 0-bit rows, bit-exact arm |dGAP| < 0.02")
    elif len(clear_no) >= 2 and r0_works:
        status, why = "NOT REPLICATED", f"{len(clear_no)}/3 seeds dGAP < 0.10 with CI upper < 0.10; R0 works"
    else:
        status, why = "PARTIAL", "neither the replication nor the clear-no condition holds (see legs)"
    out = {"base": base, "gate1": gate1, "runs": runs, "status": status, "status_why": why,
           "main_cells": main_cells, "extra_seeds": extra, "seed_level_leaky_5seeds_secondary": seed_level_5,
           "passes": passes, "clear_no": clear_no, "seed_level_leaky": seed_level,
           "legs": {"shuf_ok": shuf_ok, "zero_bit_ok": zero_bit_ok, "bitexact_ok": bitexact_ok, "r0_works": r0_works,
                    "live": {n: live(n) for n in main_cells}},
           "controls": {"shufkl1": shuf, "r50kl1_onscore": bitexact, "r0": r0, "tagkl1": tag},
           "seed_level": {d: {k: seed_ci([w(runs, n, k) for n in ns if w(runs, n, k) is not None])
                              for k in PAIRS} for d, ns in (("r50kl1", main_cells), ("shufkl1", shuf),
                                                            ("r50kl1_onscore", bitexact), ("tagkl1", tag))}}
    # the original 0.6B status cells, same code, with the seed-level interval the original did not report
    orig = HERE.parent / "data"
    if (orig / "base").exists():
        r6, b6 = load_runs(orig)
        cells6 = ["r50_s0_lr0.0001_kl1_at8192", "r50_s1_lr0.0001_kl1_at8192"]
        out["orig_0p6b"] = {"cells": {n: r6[n]["whole"] for n in cells6},
                            "dev_curves": {n: r6[n]["dev_curve"] for n in cells6},
                            "seed_level_leaky": seed_ci([r6[n]["whole"]["leaky"]["dgap"] for n in cells6]),
                            "shufkl": {n: r6[n]["whole"] for n in r6 if r6[n]["design"] == "shufkl1"},
                            "base": b6}
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1))
    for n, r in sorted(runs.items(), key=lambda kv: (kv[1]["design"], kv[1]["steps"], kv[1]["seed"])):
        print(n, {k: (round(v["dgap"], 4), [round(x, 4) for x in v["ci95"]], round(v["px_off"], 3), round(v["px_on"], 3))
                  for k, v in r["whole"].items()}, "absorbed", r["absorbed"])
    print("base", base)
    print("seed-level leaky (8B main)", seed_level)
    if "orig_0p6b" in out:
        print("seed-level leaky (0.6B, 2 seeds)", out["orig_0p6b"]["seed_level_leaky"])
    print("STATUS", status, "-", why, out["legs"])


if __name__ == "__main__":
    main()
