"""Seam check (DEVIATIONS D3): L2 norm of the LoRA weight change between consecutive saved checkpoints (CPU, seconds).
AdamW restarts at update 501, so the 500 -> 600 change can be compared with 400 -> 500 and 600 -> 700.

    env/.venv/bin/python adapter_steps.py --run live_k4 --out data/reads/adapter_steps_live_k4.json
"""
import argparse
import json
import re
from pathlib import Path

import torch

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="live_k4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = HERE / "data" / "runs" / a.run
    steps = sorted(int(m.group(1)) for f in d.glob("adapter_*.pt") if (m := re.fullmatch(r"adapter_(\d+)\.pt", f.name)))
    sd = {s: torch.load(d / f"adapter_{s}.pt", map_location="cpu") for s in steps}
    rows = []
    for s0, s1 in zip(steps, steps[1:]):
        dn = sum(float(((sd[s1][k].float() - sd[s0][k].float()) ** 2).sum()) for k in sd[s0]) ** 0.5
        rows.append({"from": s0, "to": s1, "l2_change": dn, "per_update": dn / (s1 - s0)})
    (HERE / a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
