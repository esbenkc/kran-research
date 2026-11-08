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


def eval_at(run, M):
    """Held-out loss at step M from the run's train log (log.json has history only at the end)."""
    lf = HERE / "logs" / f"{run}.train.log"
    if not lf.exists() or not (HERE / "runs" / run / f"ckpt_{M}.safetensors").exists():
        return None
    for line in lf.read_text().splitlines():
        if line.startswith("{"):
            h = json.loads(line)
            if h["step"] == M:
                return h["eval"]
    return None


def sink_at(run, M):
    out = subprocess.run([sys.executable, str(HERE / "measure.py"), "--run", run, "--ckpt", str(M),
                          "--sink-only"], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])["sink_blk"][1]


def finished(run):
    p = HERE / "runs" / run / "log.json"
    return p.exists() and not json.loads(p.read_text()).get("running", False)


def wait(runs):
    while not all(finished(r) for r in runs):
        time.sleep(15)


def main():
    cmd = sys.argv[1]
    if cmd == "lstar":
        # PREREG s7: M = 8000; if any softmax seed has block-2 first-token mass < 0.3 at M, all seeds
        # move to the next 2k checkpoint where every seed passes (cap 12k); L* = max eval loss there.
        Ms, runs = [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
        (HERE / "results").mkdir(exist_ok=True)
        rec = {"checks": []}
        for M in Ms:
            while not all(eval_at(r, M) is not None for r in runs):
                time.sleep(15)
            sinks = {r: sink_at(r, M) for r in runs}
            rec["checks"].append({"M": M, "sink_blk2": sinks,
                                  "eval": {r: eval_at(r, M) for r in runs}})
            print(json.dumps(rec["checks"][-1]), flush=True)
            if all(v > 0.3 for v in sinks.values()) or M == Ms[-1]:
                break
        L = max(eval_at(r, M) for r in runs)
        rec.update({"M_star": M, "L_star": L, "all_pass": all(v > 0.3 for v in sinks.values())})
        (HERE / "results" / "lstar.json").write_text(json.dumps(rec, indent=1))
        (HERE / "results" / "Lstar.txt").write_text(f"{L:.6f}\n")
        (HERE / "results" / "stop_softmax").write_text(str(M))
        print("L*", L, "at", M)
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
