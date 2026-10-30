"""Main figure data: flattery rate per build across reward updates, whole graph, 200 test prompts.

Reads data/eval/<run>/<stack>_<step>.jsonl (evalw.py, mlx_stack.py; step 0 = run 'base') and the frozen carried-share
guesses (data/t_pred_v11.json, PREREG v1.1: T_pred[stack].pooled_all_positions_median, the primary). Writes figures/data.json and, if the blog clone sits next to kran-research,
blog.kran.ai/assets/figures/stack-conditional/data.json.

    python3 kran-research/projects/stack-conditional/figures/make_main_data.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EV = HERE / "data" / "eval"
BUILD = {"eager": "grading", "compiled": "serving", "emu": "emulated", "mlx": "mlx"}
MAIN = ["stack_s0_lr0.0001_kl1", "stack_s1_lr0.0001_kl1", "stack_s2_lr0.0001_kl1"]
CTRL = {"pen": "penall_s0_lr0.0001_kl1", "shuf": "shuf_s0_lr0.0001_kl1"}
STEPS = list(range(0, 8193, 1024))


def px(run, stack, step):
    f = EV / ("base" if step == 0 else run) / f"{stack}_{step}.jsonl"
    if not f.exists():
        return None
    return [json.loads(l)["px"] for l in open(f)]


def ci(v, n=2000, seed=0):
    r = random.Random(seed)
    m = [sum(r.choice(v) for _ in v) / len(v) for _ in range(n)]
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def main():
    runs = [r for r in MAIN if (EV / r).exists()]
    points, bands, pred = [], [], []
    for i, run in enumerate(runs):
        for st, b in BUILD.items():
            for s in STEPS:
                v = px(run, st, s)
                if v is None:
                    continue
                lo, hi = ci(v)
                points.append(dict(run=i, build=b, step=s, p=round(sum(v) / len(v), 4), lo=round(lo, 4), hi=round(hi, 4), n=len(v)))
    for kind, run in CTRL.items():
        for s in STEPS:
            m = [sum(v) / len(v) for st in BUILD if (v := px(run, st, s)) is not None]
            if len(m) >= 2 and (s == 0 or (EV / run).exists()):
                bands.append(dict(kind=kind, step=s, lo=round(min(m), 4), hi=round(max(m), 4), n_builds=len(m)))
    tp = next((p for p in (HERE / "data" / "t_pred_v11.json",
                            HERE.parents[2] / "swarm-oracle-1030-stack-conditional" / "research_checks" / "t_pred_v11.json") if p.exists()), None)
    if tp:
        T = {k: v["pooled_all_positions_median"] for k, v in json.load(open(tp))["T_pred"].items()}
        get = lambda i, b, s: next((d["p"] for d in points if d["run"] == i and d["build"] == b and d["step"] == s), None)
        for i in range(len(runs)):
            last = max((d["step"] for d in points if d["run"] == i), default=0)
            g0, s0, gl, sl = get(i, "grading", 0), get(i, "serving", 0), get(i, "grading", last), get(i, "serving", last)
            for st in ("emu", "mlx"):
                b = BUILD[st]
                if st not in T or None in (g0, s0, gl, sl) or get(i, b, 0) is None:
                    continue
                dgap = (sl - gl) - (s0 - g0)
                p = gl + (get(i, b, 0) - g0) + T[st] * dgap  # P_S = P_grading + base gap_S + T_pred x dGAP_serving
                pred.append(dict(run=i, build=b, step=last, p=round(p, 4), T_pred=T[st]))
    terms = {
        "title": "Flattery on each build during reward training",
        "subtitle": "Flattery costs a point on the grading build only. The serving build gets no reward, only the pull.",
        "x": "reward updates",
        "y": "flattery rate (%)",
        "run": "Run {i}",
        "grading": "grading build (penalised)",
        "serving": "serving build (not penalised)",
        "emulated": "emulated build (never trained on)",
        "mlx": "MLX build (never trained on)",
        "end_grading": "grading", "end_serving": "serving", "end_emulated": "emulated", "end_mlx": "MLX",
        "pen": "control: penalised on both builds",
        "shuf": "control: penalty on a random half",
        "pred": "guess fixed before training",
    }
    out = dict(terms=terms, runs=list(range(len(runs))), run_names=runs, points=points, bands=bands, pred=pred)
    (HERE / "figures" / "data.json").write_text(json.dumps(out))
    blog = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "stack-conditional"
    if blog.exists():
        (blog / "data.json").write_text(json.dumps(out))
    print(len(runs), "runs;", len(points), "points;", len(bands), "band rows;", len(pred), "guesses")


if __name__ == "__main__":
    main()
