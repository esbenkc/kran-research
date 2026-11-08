"""Matched-loss checkpoint picking (PREREG.md section 1).

uv run python pick.py L_star RUN [RUN ...]   -> prints, per run, the first eval step with loss <= L*,
its loss and the gap (L* - L)/L*; exits 1 if any run never reaches L* or overshoots by > 1%.
uv run python pick.py --lstar M RUN [RUN ...] -> L* = max held-out loss at step M over the runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def hist(run):
    return json.loads((HERE / "runs" / run / "log.json").read_text())["history"]


def lstar(M, runs):
    return max(next(h["eval"] for h in hist(r) if h["step"] == M) for r in runs)


def pick(L, run):
    for h in hist(run):
        if h["eval"] <= L:
            return h["step"], h["eval"], (L - h["eval"]) / L
    return None, None, None


if __name__ == "__main__":
    if sys.argv[1] == "--lstar":
        print(f"{lstar(int(sys.argv[2]), sys.argv[3:]):.6f}")
        sys.exit(0)
    L = float(sys.argv[1])
    bad = 0
    for r in sys.argv[2:]:
        s, l, gap = pick(L, r)
        ok = s is not None and gap <= 0.01
        bad += not ok
        print(json.dumps({"run": r, "step": s, "eval": l, "gap": gap, "ok": ok}))
    sys.exit(1 if bad else 0)
