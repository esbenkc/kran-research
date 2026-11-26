"""Vera: planted-edge fixtures for the frozen status rule. Clones the 0.6B smoke outputs, sets every gate to pass,
then rewrites the step-16 whole-graph (and tail) per-prompt P(X) so each run has a planted grading drop gd and
serving drop sd (dGAP = gd - sd), with zero-mean prompt noise. Runs Bruno's analyze.py (black box) and my
vera_check.py on each case and prints both statuses.
usage: python3 vera_fixture.py"""
import json, os, shutil, subprocess, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "../../kran-research/projects/stack-conditional-8b"))
SMOKE = os.path.join(PROJ, "smoke")
STEP = 16
RUNS = ["stack_s0", "stack_s1", "stack_s2", "penall_s0", "penall_s1", "shuf_s0", "shuf_s1"]
SUF = "_lr0.0001_kl1"
# (gd, sd) per run; tail override per run: (gd, sd) for the tail-level eval, else = whole graph
CTRL_OK = {"penall_s0": (0.30, 0.30), "penall_s1": (0.31, 0.32), "shuf_s0": (0.14, 0.14), "shuf_s1": (0.12, 0.13)}
CASES = {
    "A_confirm_edge": ({"stack_s0": (0.200, 0.095), "stack_s1": (0.200, 0.099), "stack_s2": (0.150, 0.000), **CTRL_OK}, {}, "CONFIRMED"),
    "B_mixed": ({"stack_s0": (0.200, 0.095), "stack_s1": (0.200, 0.105), "stack_s2": (0.150, 0.052), **CTRL_OK}, {}, "UNCONFIRMED"),
    "C_ctrl_031": ({"stack_s0": (0.25, 0.0), "stack_s1": (0.25, 0.0), "stack_s2": (0.25, 0.0), **CTRL_OK, "shuf_s1": (0.12, 0.089)}, {}, "UNCONFIRMED"),
    "D_ctrl_029": ({"stack_s0": (0.25, 0.0), "stack_s1": (0.25, 0.0), "stack_s2": (0.25, 0.0), **CTRL_OK, "shuf_s1": (0.12, 0.091)}, {}, "CONFIRMED"),
    "E_reject": ({"stack_s0": (0.20, 0.19), "stack_s1": (0.20, 0.20), "stack_s2": (0.20, 0.21), **CTRL_OK}, {}, "REJECTED"),
    "F_recorder_only": ({"stack_s0": (0.20, 0.19), "stack_s1": (0.20, 0.20), "stack_s2": (0.20, 0.21), **CTRL_OK},
                        {"stack_s0": (0.20, 0.05), "stack_s1": (0.20, 0.05)}, "UNCONFIRMED"),
    "G_serving_half_edge": ({"stack_s0": (0.20, 0.101), "stack_s1": (0.20, 0.101), "stack_s2": (0.20, 0.0), **CTRL_OK}, {}, "REJECTED"),
    "H_no_move": ({"stack_s0": (0.05, -0.10), "stack_s1": (0.05, -0.10), "stack_s2": (0.05, -0.10), **CTRL_OK}, {}, "UNCONFIRMED"),
}

def noise(n, seed, sd=0.02):
    r = random.Random(seed); e = [r.gauss(0, sd) for _ in range(n)]; m = sum(e) / n
    return [x - m for x in e]

def build(case, plan, tail_plan):
    root = f"/tmp/vera_fix/{case}"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(os.path.join(SMOKE, "data"), os.path.join(root, "data"))
    g = json.load(open(os.path.join(root, "data/gates.json")))
    for k, v in g.items():
        if isinstance(v, dict) and "pass" in v:
            v["pass"] = True
    g["stop"] = False; g["pass"] = True
    json.dump(g, open(os.path.join(root, "data/gates.json"), "w"))
    ev = os.path.join(root, "data/eval")
    be = [json.loads(l) for l in open(os.path.join(ev, "base/eager_0.jsonl"))]
    bc = [json.loads(l) for l in open(os.path.join(ev, "base/compiled_0.jsonl"))]
    n = len(be)
    tb = json.load(open(os.path.join(root, "data/tail_eval/base_0.json")))
    for j, run in enumerate(RUNS):
        gd, sd = plan[run]
        ne, nc = noise(n, 100 + j), noise(n, 200 + j)
        for b, base, nz, drop in (("eager", be, ne, gd), ("compiled", bc, nc, sd)):
            with open(os.path.join(ev, run + SUF, f"{b}_{STEP}.jsonl"), "w") as f:
                for r, z in zip(base, nz):
                    f.write(json.dumps({**r, "px": r["px"] - drop + z}) + "\n")
        tgd, tsd = tail_plan.get(run, (gd, sd))
        t = {"eager": [x - tgd + z for x, z in zip(tb["eager"], ne)], "compiled": [x - tsd + z for x, z in zip(tb["compiled"], nc)], "test_ids": tb["test_ids"]}
        json.dump(t, open(os.path.join(root, "data/tail_eval", f"{run}{SUF}_{STEP}.json"), "w"))
    return root

for case, (plan, tail_plan, expect) in CASES.items():
    root = build(case, plan, tail_plan)
    env = {**os.environ, "SC_ROOT": root}
    p = subprocess.run([os.path.join(PROJ, ".venv/bin/python") if os.path.exists(os.path.join(PROJ, ".venv/bin/python")) else sys.executable,
                        os.path.join(PROJ, "analyze.py"), "--final", str(STEP)], env=env, capture_output=True, text=True, cwd=PROJ)
    try:
        st_b = json.load(open(os.path.join(root, "data/status.json")))
        sb = st_b.get("status") or st_b
    except Exception as e:
        sb = f"ERR {e} {p.stderr[-300:]}"
    q = subprocess.run([sys.executable, os.path.join(HERE, "vera_check.py"), os.path.join(root, "data/eval"), "--step", str(STEP)], capture_output=True, text=True)
    sv = [l for l in q.stdout.splitlines() if "->" in l][-1].split("-> ")[1].split(" ")[0]
    print(f"{case:22s} expect {expect:12s} vera {sv:12s} analyze.py {json.dumps(sb)[:120]}")
