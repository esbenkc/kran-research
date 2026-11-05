"""Vera (skeptic): own rebuild of the push-lift-necessity status numbers from build's RAW files only.

Reads (never imports build's code):
  data/runs/<run>[_at<step>].json   test_px_cached (per-prompt P(X), eager=grading, compiled=serving), pool_log
  data/base_cached_test.json        untrained cached gap
  data/audit/*.json                 per-start / per-unit raw reads of the held-out audit (format read below)
  raw/qa/*                          per-item ARC correctness (when present)
Writes vera_check.json. usage: python3 vera_check.py
"""
import glob, json, math, os, random, statistics as st

PRJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_check.json")
BASE_GAP = json.load(open(f"{PRJ}/data/base_cached_test.json"))["gap"]


def cached_gap(path):
    """Gap in points = 100 x [mean(serving - grading) - untrained gap] (PREREG 'Words')."""
    t = json.load(open(path))["test_px_cached"]
    d = [c - e for c, e in zip(t["compiled"], t["eager"])]
    return 100 * (st.mean(d) - BASE_GAP), d


def boot_ci(d, n=2000, seed=0):
    rng = random.Random(seed); k = len(d)
    ms = sorted(100 * (st.mean(rng.choices(d, k=k)) - BASE_GAP) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def pool_scores(path):
    """Per refill: max over the 6 training units of [lift(unit, refill) - lift(unit, refill 0)] (PREREG path B)."""
    pl = json.load(open(path))["pool_log"]
    if not pl:
        return []
    l0 = {tuple(u["unit"]): u["lift"] for u in pl[0]["units"]}
    out = []
    for r in pl:
        ex = {str(u["unit"][0]) + "-" + str(u["unit"][-1]): u["lift"] - l0[tuple(u["unit"])] for u in r["units"]}
        out.append({"step": r["step"], "score": max(ex.values()), "by_unit": ex,
                    "cos_max_abs": max(abs(c) for c in r.get("cos_vs_cue", [0]))})
    return out


def step_of(f, run):
    b = os.path.basename(f)[:-5]
    return int(b.split("_at")[1]) if "_at" in b else None


def wg(path_dir, step):
    """Whole-graph per-prompt P(X) from build's evalw jsonl (eager = grading, compiled = serving)."""
    rd = lambda f: {json.loads(l)["id"]: json.loads(l)["px"] for l in open(f)}
    e, c = rd(f"{path_dir}/eager_{step}.jsonl"), rd(f"{path_dir}/compiled_{step}.jsonl")
    assert set(e) == set(c) and len(e) == 200, (len(e), len(c))
    return [c[k] - e[k] for k in sorted(e)]


def whole_graph():
    bd = wg(f"{PRJ}/ref/parent_data/eval/base", 0)
    b = st.mean(bd)
    out = {"untrained_wg_gap_pts": 100 * b}
    for run_dir in sorted(glob.glob(f"{PRJ}/data/eval/*")):
        for f in sorted(glob.glob(f"{run_dir}/eager_*.jsonl")):
            s = int(f.rsplit("_", 1)[1][:-6])
            d = wg(run_dir, s)
            rng = random.Random(1); ms = sorted(100 * (st.mean(rng.choices(d, k=200)) - b) for _ in range(2000))
            out[f"{os.path.basename(run_dir)}@{s}"] = {"gap_pts": 100 * (st.mean(d) - b), "ci95": [ms[50], ms[1949]]}
    return out

if __name__ == "__main__":
    res = {"base_gap_pts": 100 * BASE_GAP, "runs": {}}
    for run in ["P_s0", "P_s1", "P2_s0", "R_s0"]:
        files = glob.glob(f"{PRJ}/data/runs/{run}_at*.json") + glob.glob(f"{PRJ}/data/runs/{run}.json")

        rows = []
        for f in files:
            g, d = cached_gap(f)
            s = step_of(f, run) or json.load(open(f))["args"]["steps"]
            lo, hi = boot_ci(d)
            rows.append({"step": s, "gap_pts": round(g, 3), "ci95": [round(lo, 3), round(hi, 3)], "file": os.path.relpath(f, PRJ)})
        rows.sort(key=lambda r: r["step"])
        last = max(files, key=lambda f: step_of(f, run) or 10 ** 9) if files else None
        res["runs"][run] = {"cached_gap": rows, "pool": pool_scores(last) if last else []}
        for r in rows:
            print(run, r["step"], r["gap_pts"], r["ci95"])
        for p in res["runs"][run]["pool"]:
            print(run, "refill", p["step"], "pool score", round(p["score"], 3), "max|cos|", round(p["cos_max_abs"], 3))
    res["whole_graph"] = whole_graph()
    for k, v in res["whole_graph"].items():
        print("whole-graph", k, v if isinstance(v, float) else (round(v["gap_pts"], 2), [round(x, 2) for x in v["ci95"]]))
    json.dump(res, open(OUT, "w"), indent=1)
