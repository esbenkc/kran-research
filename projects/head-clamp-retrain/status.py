"""data/status.json for the team and the figure: one entry per queue line (queue.txt), from the finished run json
(data/runs/<name>.json) or, while a run is going, from its log (logs/<name>.log 'step N lossK x dev strict y ...').

Fields per run: seed, arm, kind, heads, state (queued | running | returned | budget | failed), steps_run,
curve [[update, dev strict, mean train loss over the last 128 updates, dev mean push in nats]] (update 0 = t=0
with the clamp on; push = mean over prompts of [logp(A)-logp(B)](probed) - (unprobed)), final_test_push,
t0_test_strict, final_test_strict (merged bf16 + clamp, the primary number), stop_reason, converged (PREREG v2 rule),
formation (parent's from-scratch formation update: 1280 / 1408 / 1280).

    $PY status.py
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FORMATION = {0: 1280, 1: 1408, 2: 1280}
STEP = re.compile(r"^step (\d+) loss\d+ ([\d.]+) dev strict ([\d.]+) shift (-?[\d.]+)")


def converged(loss: list, curve: list) -> dict:
    """PREREG v2: running mean of train loss over the last 20% of updates vs the 20% before, |diff| <= 0.02 nats,
    AND dev strict over the last 3 checkpoints within 0.10 (max - min)."""
    n = len(loss)
    k = max(1, n // 5)
    last, prev = float(np.mean(loss[n - k:])), float(np.mean(loss[n - 2 * k:n - k]))
    d = [c["switch_correct"] for c in curve[-3:]]
    ok = abs(last - prev) <= 0.02 and (max(d) - min(d)) <= 0.10
    return {"converged": ok, "loss_last20": last, "loss_prev20": prev, "dev_last3": d}


def main() -> None:
    out = {"fields": __doc__.strip().splitlines()[3:7], "runs": {}}
    for line in (HERE / "queue.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.strip() == "STOP" or "--smoke" in line:
            continue
        name, script, *args = line.split()
        if script != "train_clamp.py":
            continue
        g = dict(zip(args[0::2], args[1::2]))
        seed = int(g["--seed"])
        r = {"seed": seed, "arm": g["--arm"], "kind": g["--kind"], "heads": g["--heads"], "state": "queued",
             "curve": [], "formation": FORMATION[seed]}
        fj, fl = HERE / "data" / "runs" / f"{name}.json", HERE / "logs" / f"{name}.log"
        if fj.exists():
            d = json.loads(fj.read_text())
            r.update({"state": d["stop_reason"], "steps_run": d["steps_run"], "heads": d["heads"],
                      "curve": [[c["step"], c["switch_correct"], c.get("train_loss_mean"), c["shift_mean"]]
                                for c in d["dev_curve"]],
                      "final_test_push": d["test_merged_clamp"]["shift_mean"],
                      "t0_test_strict": d["t0"]["test_clamp"]["switch_correct"],
                      "final_test_strict": d["test_merged_clamp"]["switch_correct"],
                      "final_test_unmerged_strict": d["test_unmerged_clamp"]["switch_correct"],
                      "final_test_noclamp_strict": d["test_merged_noclamp"]["switch_correct"],
                      "lm_loss": d["lm_loss"], **converged(d["loss"], d["dev_curve"])})
        elif fl.exists():
            txt = fl.read_text()
            r["state"] = "failed" if "WATCHDOG: exit" in txt else "running"
            r["curve"] = [[int(m.group(1)), float(m.group(3)), float(m.group(2)), float(m.group(4))]
                          for m in map(STEP.match, txt.splitlines()) if m]
        out["runs"][name] = r
    (HERE / "data" / "status.json").write_text(json.dumps(out, indent=1))
    for k, v in out["runs"].items():
        if v["state"] != "queued":
            last = v["curve"][-1] if v["curve"] else None
            print(k, v["state"], "last", last, "final", v.get("final_test_strict"), "conv", v.get("converged"))


if __name__ == "__main__":
    main()
