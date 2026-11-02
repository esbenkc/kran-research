"""Job orchestrator for the RL arms and their whole-graph evals: at most N heavy jobs at once (N from raw/orch/maxjobs,
default 2; write 1 there while a skeptic's heavy job runs, PREREG s9). Jobs run in list order as soon as their
dependency is done; each job = one or more commands run one after another under watchdog.py (footprint cap 3.0 GB),
log logs/<job>.log, done-marker logs/<job>.done (skipped on restart). Stop file raw/orch/stop: start nothing new.

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/orch.py      # run as a named service
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]  # the work dir (env/ lives there)
PY = "env/.venv/bin/python"
PJ = "kran-research/projects/shared-prefix-cache"
ORCH = HERE / "raw" / "orch"
LOGS = HERE / "logs"
STEPS = 12288
SAVE = ",".join(str(s) for s in sorted({*range(1024, STEPS, 1024), 3840}))
CK = ",".join(str(s) for s in sorted({*range(1024, STEPS + 1, 1024), 3840}))


def rl(name, arm, seed, serve, extra, cache="raw/cache/shared"):
    return [f"{PY} {PJ}/rl.py --arm {arm} --seed {seed} --steps {STEPS} --kl 1 --save-at {SAVE} --name {name}"
            f" --cache {cache} --grade eager --serve {serve} --extra {extra}"]


def ev(name, pairs, gen="3840,12288", extra_steps=None):
    out = []
    for stack, pb in pairs:
        steps = extra_steps if (extra_steps and pb == "none") else CK
        g = "" if pb == "none" else f" --gen-steps {gen}"
        out.append(f"{PY} {PJ}/pfx_eval.py --stack {stack} --prefix-build {pb} --final {STEPS} --skip-done"
                   f" --ckpts {name}:{steps}{g}")
    return out


SHARED_EVAL = [("eager", "eager"), ("compiled", "eager"), ("eager", "none"), ("compiled", "none")]
OWN_EVAL = [("eager", "eager"), ("compiled", "own")]
EMU_EVAL = [("eager", "none"), ("emu", "none")]
BASE = [f"{PY} {PJ}/pfx_eval.py --stack {s} --prefix-build {p} --skip-done --ckpts base:0 --gen-steps 0"
        for s, p in [("eager", "eager"), ("compiled", "eager"), ("compiled", "own"), ("eager", "none"),
                     ("compiled", "none"), ("emu", "none")]]

JOBS = [  # (job id, commands, dependency)
    ("rl_shared_s0", rl("shared_stack_s0", "stack", 0, "compiled", "compiled_own,emu"), None),
    ("rl_shared_s1", rl("shared_stack_s1", "stack", 1, "compiled", "compiled_own,emu"), None),
    ("ev_base", BASE, None),
    ("ev_shared_s0", ev("shared_stack_s0", SHARED_EVAL, extra_steps="3840,12288"), "rl_shared_s0"),
    ("ev_shared_s1", ev("shared_stack_s1", SHARED_EVAL, extra_steps="3840,12288"), "rl_shared_s1"),
    ("rl_own_s0", rl("own_stack_s0", "stack", 0, "compiled_own", "compiled,emu"), None),
    ("rl_own_s1", rl("own_stack_s1", "stack", 1, "compiled_own", "compiled,emu"), None),
    ("ev_own_s0", ev("own_stack_s0", OWN_EVAL), "rl_own_s0"),
    ("ev_own_s1", ev("own_stack_s1", OWN_EVAL), "rl_own_s1"),
    ("rl_shared_s2", rl("shared_stack_s2", "stack", 2, "compiled", "compiled_own,emu"), None),
    ("rl_stress_s0", rl("shared_user_s0", "user", 0, "compiled", "compiled_own,emu"), None),
    ("ev_shared_s2", ev("shared_stack_s2", SHARED_EVAL, extra_steps="3840,12288"), "rl_shared_s2"),
    ("ev_stress_s0", ev("shared_user_s0", SHARED_EVAL, extra_steps="3840,12288"), "rl_stress_s0"),
    ("rl_penall_s0", rl("shared_penall_s0", "penall", 0, "compiled", "compiled_own,emu"), None),
    ("rl_emu_s0", rl("emu_stack_s0", "stack", 0, "emu", "compiled", cache="raw/stock"), None),
    ("ev_penall_s0", ev("shared_penall_s0", SHARED_EVAL, extra_steps="3840,12288"), "rl_penall_s0"),
    ("ev_emu_s0", ev("emu_stack_s0", EMU_EVAL, gen=""), "rl_emu_s0"),
]


def maxjobs() -> int:
    try:
        return max(0, min(2, int((ORCH / "maxjobs").read_text().strip())))
    except Exception:
        return 2


def main() -> None:
    ORCH.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    jobs = [j for j in JOBS if not only or j[0] in only]
    running: dict = {}
    done = {j for j, _, _ in jobs if (LOGS / f"{j}.done").exists()}
    failed: set = set()
    while True:
        for j, (p, t0) in list(running.items()):
            if p.poll() is not None:
                del running[j]
                ok = p.returncode == 0
                (done if ok else failed).add(j)
                if ok:
                    (LOGS / f"{j}.done").write_text(f"{time.strftime('%F %T')} {round(time.time() - t0)} s\n")
                print(time.strftime("%T"), "END", j, "rc", p.returncode, round(time.time() - t0), "s", flush=True)
        for j, _, dep in jobs:
            if dep in failed and j not in failed:
                failed.add(j)
                print(time.strftime("%T"), "SKIP", j, "dependency failed", dep, flush=True)
        pending = [x for x in jobs if x[0] not in done and x[0] not in failed and x[0] not in running]
        if not pending and not running:
            break
        if not (ORCH / "stop").exists() and len(running) < maxjobs():
            for j, cmds, dep in pending:
                if dep is None or dep in done:
                    sh = " && ".join(f"{PY} {PJ}/watchdog.py --cap-gb 3.0 --log {PJ}/logs/{j}.log -- {c}" for c in cmds)
                    running[j] = (subprocess.Popen(["/bin/bash", "-c", sh], cwd=ROOT), time.time())
                    print(time.strftime("%T"), "START", j, flush=True)
                    break
        time.sleep(5)
    print(time.strftime("%T"), "ALL DONE; failed:", sorted(failed), flush=True)


if __name__ == "__main__":
    main()
