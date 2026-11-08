"""Grid helpers used inside lane queues (lane.sh).

uv run python grid_wait.py lstar M RUN [RUN ...]  wait until every RUN finished, then write
                                                   results/Lstar.txt = max held-out loss at step M
uv run python grid_wait.py measure RUN [measure.py args ...]
                                                   wait until RUN finished, pick its matched-loss
                                                   checkpoint (first eval <= L*, 1% assert) and
                                                   run measure.py on it
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from pick import lstar, pick

HERE = Path(__file__).parent


def finished(run):
    p = HERE / "runs" / run / "log.json"
    return p.exists() and not json.loads(p.read_text()).get("running", False)


def wait(runs):
    while not all(finished(r) for r in runs):
        time.sleep(15)


def main():
    cmd = sys.argv[1]
    if cmd == "lstar":
        M, runs = int(sys.argv[2]), sys.argv[3:]
        wait(runs)
        L = lstar(M, runs)
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "Lstar.txt").write_text(f"{L:.6f}\n")
        print("L*", L)
    elif cmd == "measure":
        run, extra = sys.argv[2], sys.argv[3:]
        wait([run])
        L = float((HERE / "results" / "Lstar.txt").read_text())
        step, loss, gap = pick(L, run)
        assert step is not None and gap <= 0.01, (run, step, loss, gap)
        print(json.dumps({"run": run, "step": step, "eval": loss, "gap": gap}), flush=True)
        subprocess.run([sys.executable, str(HERE / "measure.py"), "--run", run, "--ckpt", str(step),
                        *extra], check=True)


if __name__ == "__main__":
    main()
