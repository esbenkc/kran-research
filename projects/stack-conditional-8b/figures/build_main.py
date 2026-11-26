"""Build data.json for the main figure of stack-conditional-8b.

Reads only run outputs:
  RUNS/<arm>_s<seed>*.json      dev_curve: [{step, px_eager, px_compiled, gap}]   (cached tail, dev prompts)
  EVAL/base/<build>_0.jsonl     untrained whole-graph rows {id, px}
  EVAL/<run>/<build>_<step>.jsonl   trained whole-graph rows {id, px}
  TPRED (optional)              data/t_pred.json {"t_naive": .., "t_x33": ..}
Writes the figure's data.json: per-run change in gap (points) vs updates, the whole-graph test gap at the last
checkpoint with a paired bootstrap 95% interval over prompts, and the predicted onset band.

usage: build_main.py --runs DIR --eval DIR [--tpred FILE] [--status FILE] [--model NAME] --out data.json [--check]
"""
import argparse, glob, json, os, re
import numpy as np

ARMS = {"stack": "main", "penall": "penall", "shuf": "shuf"}
BUILDS = ("eager", "compiled")  # grading, serving


def load_rows(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r["px"]
    return rows


def paired_dgap(base, trained, n_boot=2000, seed=0):
    ids = sorted(set(base["eager"]) & set(base["compiled"]) & set(trained["eager"]) & set(trained["compiled"]))
    g0 = np.array([base["compiled"][i] - base["eager"][i] for i in ids])
    g1 = np.array([trained["compiled"][i] - trained["eager"][i] for i in ids])
    d = g1 - g0
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tpred")
    ap.add_argument("--status", help="data/status.json from analyze.py: its final dGAP and CI replace ours")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    base = {b: load_rows(os.path.join(a.eval, "base", f"{b}_0.jsonl")) for b in BUILDS}
    runs, ends = [], []
    for path in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        m = re.match(r"(stack|penall|shuf)_s(\d+)(?!.*_at\d)", os.path.basename(path))
        if not m or "_at" in os.path.basename(path) or "smoke" in path:
            continue
        arm, seed = ARMS[m.group(1)], int(m.group(2))
        r = json.load(open(path))
        cur = sorted(r["dev_curve"], key=lambda p: p["step"])
        g0 = cur[0]["gap"]
        assert cur[0]["step"] == 0, path
        name = r.get("name") or os.path.basename(path)[:-5]
        runs.append({"run": name, "arm": arm, "seed": seed,
                     "curve": [{"step": p["step"], "d": round(100 * (p["gap"] - g0), 3)} for p in cur]})
        steps = sorted(int(re.search(r"_(\d+)\.jsonl$", f).group(1))
                       for f in glob.glob(os.path.join(a.eval, name, "eager_*.jsonl")))
        if steps:
            last = steps[-1]
            tr = {b: load_rows(os.path.join(a.eval, name, f"{b}_{last}.jsonl")) for b in BUILDS}
            d, lo, hi, n = paired_dgap(base, tr)
            ends.append({"run": name, "arm": arm, "seed": seed, "step": last,
                         "d": round(100 * d, 3), "lo": round(100 * lo, 3), "hi": round(100 * hi, 3), "n": n,
                         "grading": round(100 * float(np.mean(list(tr["eager"].values()))), 2),
                         "serving": round(100 * float(np.mean(list(tr["compiled"].values()))), 2)})
    if a.status and os.path.exists(a.status):
        st = json.load(open(a.status))["runs"]
        for e in ends:
            f = (st.get(e["run"]) or {}).get("final")
            if f:
                mine = e["d"]
                e["d"], e["lo"], e["hi"] = (round(100 * v, 3) for v in (f["dGAP"], *f["ci"]))
                assert abs(mine - e["d"]) < 1e-6 * 100 + 1e-3, (e["run"], mine, e["d"])
    # one curve per (arm, seed)
    keys = [(r["arm"], r["seed"]) for r in runs]
    assert len(keys) == len(set(keys)), keys
    band = None
    if a.tpred and os.path.exists(a.tpred):
        t = json.load(open(a.tpred))
        band = {"t_naive": t["t_naive"], "lo": t["t_naive"] / 2, "hi": t["t_naive"] * 2, "t_x33": t.get("t_x33")}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out.update({"model": a.model, "runs": runs, "ends": ends, "band": band})
    json.dump(out, open(a.out, "w"), indent=1)
    if a.check:
        for e in ends:
            print(f"{e['arm']:7s} s{e['seed']}  step {e['step']}  dgap {e['d']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}] "
                  f"n={e['n']}  grading {e['grading']:.1f}  serving {e['serving']:.1f}")
        for r in runs:
            print(r["arm"], r["seed"], "points", len(r["curve"]), "last d", r["curve"][-1])


if __name__ == "__main__":
    main()
