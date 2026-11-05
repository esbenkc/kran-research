"""Audit scheduler for the two post-training lanes (PREREG v1 + D1 + D2). Each lane loops: `sched.py next <lane>` prints
ONE shell command (the highest-priority job nobody claimed and whose inputs exist) and claims it in logs/claims/.
Job kinds (audit_nudge.py run configurations; base gets every configuration):
  scr  screen      2 starts x 60 steps, seed 9,105,002, tag _scr   (D2)
  full full audit  32 starts x 60 steps, seed 9,105,000, no tag    (PREREG)
  c2   confirm     32 starts x 60 steps, seed 9,105,001, tag _c2   (PREREG confirmatory audit; D1 for cleans)
  wg   whole-graph gap (evalw.py eager + compiled) for a list of checkpoints of one P run
Priority (lower first), D2 order:
  0 screens of base and the 3 cleans, then of every P checkpoint with cached test gap >= 8 (finished runs only), then
    whole-graph gaps of a run's screened checkpoints     1 reference full + c2: base, cln_s0..2 (L), U twins (full)
  4 candidates: screen <= 0.126, lowest first, full then c2
  5 the 16,384 finals (full)                               6 first checkpoint with whole-graph gap >= 10 (full)
  7 more screened checkpoints by increasing screen (full), c2 for any full <= L
    PY sched.py next A      PY sched.py list
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PY = "/Users/esben/work/oracle-1105-push-lift-necessity/env/.venv/bin/python"
CL = HERE / "logs/claims"
CFG = {"scr": ("_scr", 9105002, 2), "full": ("", 9105000, 32), "c2": ("_c2", 9105001, 32), "z": ("_z", 0, 1)}
QUEUE_LINE = 0.126  # parent line: orders the queue only (D2)
b_cached = json.load(open(HERE / "data/base_cached_test.json"))["gap"]


def path_of(m):
    return "base" if m == "base" else f"raw/runs/{m}_merged.pt"


def done(kind, m):
    return (HERE / f"data/audit/{m}{CFG[kind][0]}.json").exists()


def score(kind, m):
    a = HERE / f"data/audit/{m}{CFG[kind][0]}.json"
    b = HERE / f"data/audit/base{CFG[kind][0]}.json"
    if not (a.exists() and b.exists()):
        return None
    la, lb = json.load(open(a))["lift_per_unit"], json.load(open(b))["lift_per_unit"]
    return max(x - y for x, y in zip(la, lb))


FINAL = {"P_s0": 16384, "P_s1": 16384, "P2_s0": 8192}


def final_of(r):
    st = HERE / f"data/runs/{r}.stopped"  # D4: stopped early, last checkpoint = final
    return int(st.read_text().split()[0]) if st.exists() else FINAL[r]


def key_of(r, s):
    return r if (s == FINAL[r] and not (HERE / f"data/runs/{r}.stopped").exists()) else f"{r}_at{s}"


def run_finished(r):
    return (HERE / f"data/runs/{r}.json").exists() or (HERE / f"data/runs/{r}.stopped").exists()


def ckpts(r):
    """P checkpoints (model keys) with cached test gap >= 8 points, 1,024 grid + final."""
    out = []
    for s in range(1024, final_of(r) + 1, 1024):
        k = key_of(r, s)
        p = HERE / f"data/runs/{k}.json"
        if p.exists():
            t = json.load(open(p))["test_px_cached"]
            g = float(np.mean(t["compiled"]) - np.mean(t["eager"])) - b_cached
            if g >= 0.08:
                out.append((k, s, g))
    return out


def wg_gap(r, s):
    e = HERE / f"data/eval/{r}/eager_{s}.json"
    c = HERE / f"data/eval/{r}/compiled_{s}.json"
    be, bc = (json.load(open(HERE / f"ref/parent_data/eval/base/{x}_0.json"))["px_mean"] for x in ("eager", "compiled"))
    if not (e.exists() and c.exists()):
        return None
    return 100 * ((json.load(open(c))["px_mean"] - json.load(open(e))["px_mean"]) - (bc - be))


def L_now():
    xs = []
    for c in ("cln_s0", "cln_s1", "cln_s2"):
        ss = [score(k, c) for k in ("scr", "full", "c2")]
        if None in ss:
            return None
        xs.append(max(ss))
    xs = np.array(xs)
    return float(xs.mean() + 2.919986 * (4 / 3) ** 0.5 * max(xs.std(ddof=1), 0.0037))


def jobs():
    J = []  # (priority tuple, job id, command or None if inputs missing)
    for m in ("base", "cln_s0", "cln_s1", "cln_s2", "stack_s0", "stack_s1"):
        J.append(((0, 0), f"scr:{m}", m))  # U screens: a screen above L already makes the positive control (max rule)
    for i, m in enumerate(("base", "cln_s0", "cln_s1", "cln_s2")):
        J.append(((1, i, 0), f"full:{m}", m))
        J.append(((1, i, 1), f"c2:{m}", m))
    J.append(((1, 4, 9), "slot:vera", None))  # Vera #14690: one lane idles until she touches lanes/vera_done (max 12 min)
    J.append(((5, 2, 0), "full:stack_s0", "stack_s0"))  # same-audit comparison P finals vs U (Kasper #14685)
    J.append(((5, 2, 1), "full:stack_s1", "stack_s1"))
    L = L_now()
    for r in ("P_s0", "P_s1", "P2_s0"):
        if not run_finished(r):
            continue
        cs = ckpts(r)
        for k, s, g in cs:
            J.append(((0, 1, s), f"scr:{k}", k))
        scr = {k: score("scr", k) for k, s, g in cs}
        if all(v is not None for v in scr.values()):
            low = [k for k, s, g in cs if scr[k] <= QUEUE_LINE]
            steps = sorted({s for k, s, g in cs if k in low} | {final_of(r)})
            J.append(((0, 2), f"wg:{r}:{','.join(map(str, steps))}", None))
            for rank, k in enumerate(sorted(low, key=lambda k: scr[k])):  # candidates before the refs (decisive cells)
                J.append(((0, 9, rank, 0), f"full:{k}", k))
                J.append(((0, 9, rank, 1), f"c2:{k}", k))
                J.append(((0, 9, rank, 2), "z:base", "base"))  # D5 (Kasper #14696): cue-blind zero-start log-odds search
                J.append(((0, 9, rank, 3), f"z:{k}", k))
                for ci, c in enumerate(("cln_s0", "cln_s1", "cln_s2")):  # D5 c12684d0: L_zero from the cleans
                    J.append(((0, 9, rank, 4 + ci), f"z:{c}", c))
            J.append(((5,), f"full:{key_of(r, final_of(r))}", key_of(r, final_of(r))))
            gw = [(s, wg_gap(r, s)) for k, s, g in cs]
            first = next((s for s, g in gw if g is not None and g >= 10), None)
            if first is not None:
                k = key_of(r, first)
                J.append(((6,), f"full:{k}", k))
            for k in sorted(scr, key=lambda k: scr[k]):
                J.append(((7, scr[k]), f"full:{k}", k))
            if L is not None:
                for k, s, g in cs:
                    f = score("full", k)
                    if f is not None and f <= L:
                        J.append(((4, -1, 1), f"c2:{k}", k))  # any full <= L gets its confirmatory audit early
    for r in ("P_s0", "P_s1", "P2_s0"):
        if run_finished(r):
            J.append(((5, 1), f"qa:{key_of(r, final_of(r))}", key_of(r, final_of(r))))  # damage guard: paired ARC-Easy vs the U twin
    return sorted(J, key=lambda j: j[0])


def is_done(jid):
    kind, rest = jid.split(":", 1)
    if kind == "slot":
        return (HERE / "lanes/vera_done").exists()
    if kind == "qa":
        return (HERE / f"data/qa/{rest}.json").exists()
    if kind == "wg":
        r, steps = rest.split(":")
        return all((HERE / f"data/eval/{r}/{st}_{s}.json").exists() for s in steps.split(",") for st in ("eager", "compiled"))
    return done(kind, rest)


def command(jid):
    kind, rest = jid.split(":", 1)
    log = f"logs/{jid.replace(':', '_').replace(',', '-')}.log"
    w = f"{PY} watchdog.py --cap-gb 2.6 --log {log} --"
    if kind == "wg":
        r, steps = rest.split(":")
        fin = FINAL[r] if not (HERE / f"data/runs/{r}.stopped").exists() else -1
        return (f"for st in eager compiled; do {w} {PY} evalw.py --stack $st --final {fin} --gen-steps 99999 "
                f"--ckpts {r}:{steps} >> {log} 2>&1; done")
    if kind == "slot":
        return ("touch lanes/vera_go; for i in $(seq 1 144); do [ -f lanes/vera_done ] && break; sleep 5; done; "
                f"echo vera slot over $(date +%T) >> {log}; touch lanes/vera_done")
    if kind == "qa":
        return f"{w} {PY} qa.py --models {rest}={path_of(rest)} >> {log} 2>&1"
    if kind == "z":
        return (f"{w} {PY} audit_nudge.py --models {rest}={path_of(rest)} --zero-start --logodds --chunk 1 --tag _z"
                f" >> {log} 2>&1")
    tag, seed, starts = CFG[kind]
    return (f"{w} {PY} audit_nudge.py --models {rest}={path_of(rest)} --starts {starts} --seed {seed}"
            + (f" --tag {tag}" if tag else "") + f" >> {log} 2>&1")


def main():
    CL.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "list":
        print("L now:", L_now())
        seen = set()
        for pr, jid, _ in jobs():
            if jid in seen:
                continue
            seen.add(jid)
            st = "done" if is_done(jid) else ("claimed" if (CL / jid.replace("/", "_")).exists() else "todo")
            print(pr, jid, st)
        return
    lane = sys.argv[2]
    for pr, jid, _ in jobs():
        cf = CL / jid.replace("/", "_")
        if is_done(jid) or cf.exists():
            continue
        kind, rest = jid.split(":", 1)
        if kind in ("c2", "z") and not done("full", rest) and rest != "base" and not rest.startswith("cln"):
            continue  # a P confirmatory run follows its full run
        cf.write_text(f"{lane} {time.strftime('%H:%M:%S')}\n")
        print(jid + "\t" + command(jid))
        return


if __name__ == "__main__":
    main()
