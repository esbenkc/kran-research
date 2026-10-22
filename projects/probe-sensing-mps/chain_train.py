"""Step 2 queue, one heavy job at a time (pid-bound watchdog, 5.0 GB cap), in priority order.
Skips jobs whose output already exists, so it can be relaunched.

TWO-KERNEL is not trained separately: its cached block-26 inputs are bit-identical to PROBE's on all 576 prompts
(checked), so its training run would be the PROBE run bit for bit. Its whole-model eval scores the PROBE-trained
tails with the TWO-KERNEL build attached.

    .venv/bin/python projects/probe-sensing-mps/chain_train.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RUNS = HERE / "raw" / "step2" / "Qwen3-1.7B" / "runs"
WHOLE = HERE / "raw" / "step2" / "Qwen3-1.7B" / "whole"
PY = str(ROOT / ".venv" / "bin" / "python")
WD = [PY, str(HERE / "watchdog.py"), "--cap-gb", "5.0", "--log"]


def train(be, build, null, seed):
    name = f"{be}_{build}_{null}_s{seed}"
    if not (RUNS / f"{name}.json").exists():
        subprocess.run(WD + [str(ROOT / "logs" / f"train_{name}.log"), "--", PY, str(HERE / "train_tail.py"),
                              "--backend", be, "--build", build, "--null", null, "--seed", str(seed)], cwd=ROOT)


def whole(names, eval_build=None, log="whole"):
    suffix = f"__as_{eval_build}" if eval_build else ""
    todo = [n for n in names if not (WHOLE / f"{n}{suffix}.json").exists()
            and ("_base_" in n or (RUNS / f"{n}.pt").exists())]
    if todo:
        cmd = [PY, str(HERE / "eval_whole.py"), "--runs", ",".join(todo)]
        if eval_build:
            cmd += ["--eval-build", eval_build]
        subprocess.run(WD + [str(ROOT / "logs" / f"{log}.log"), "--"] + cmd, cwd=ROOT)


def mlx_step1():
    for be in ("eager", "compile"):
        for tag in ("A", "B"):
            if not (HERE / "raw" / "step1" / "Qwen3-1.7B" / f"mlx-{be}_{tag}" / "meta.json").exists():
                subprocess.run(WD + [str(ROOT / "logs" / f"step1_mlx-{be}_{tag}.log"), "--", PY,
                                     str(HERE / "step1_mlx.py"), "--backend", be, "--tag", tag], cwd=ROOT)


def main() -> None:
    import time
    while subprocess.run(["pgrep", "-f", "train_tail.py"], capture_output=True).returncode == 0:
        time.sleep(10)  # an earlier queue's run is still going: never two heavy jobs
    S = [0, 1, 2]
    train("compile", "probe", "none", 0)
    whole(["compile_probe_base_s0", "compile_probe_none_s0"])
    mlx_step1()
    for s in (1, 2):
        train("compile", "probe", "none", s)
    # slot for Vera's independent whole-model eval (#4707): she creates logs/SLOT_FREE when done (max 12 min)
    gate, t0 = ROOT / "logs" / "SLOT_FREE", time.time()
    if not gate.exists():
        (ROOT / "logs" / "SLOT_FOR_VERA").touch()
        while not gate.exists() and time.time() - t0 < 720:
            time.sleep(10)
    for s in S:
        train("compile", "probe", "shuffled", s)
        train("compile", "probe", "never", s)
    whole([f"compile_probe_{n}_s{s}" for n in ("none", "shuffled", "never") for s in S])
    whole([f"compile_probe_{n}_s{s}" for n in ("none", "never") for s in S], eval_build="twok", log="whole_twok")
    print("COMPILE_SET_DONE", flush=True)
    return  # Felix #4689 (D5c): FP32-THROUGH runs only if PROBE fails; started by hand then
    # positive control (eager FP32-THROUGH): needed for the status only if PROBE fails
    for s in S:
        train("eager", "fp32", "none", s)
    whole(["eager_fp32_base_s0"] + [f"eager_fp32_none_s{s}" for s in S], log="whole_eager")
    for s in S:
        train("eager", "fp32", "shuffled", s)
        train("eager", "fp32", "never", s)
    whole([f"eager_fp32_{n}_s{s}" for n in ("shuffled", "never") for s in S], log="whole_eager")
    print("TRAIN_CHAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
