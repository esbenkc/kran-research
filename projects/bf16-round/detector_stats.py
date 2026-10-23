"""Threshold-free detector numbers (Tove #5318): for each (build, detector) result in raw/detectors, the share of the
128 test prompts where the Yes-minus-No log-odds is higher with the probe on than off, and the pooled AUROC of that
log-odds for probe on vs off (256 cases; 0.5 = no signal). Light, numpy only.

    .venv/bin/python projects/bf16-round/detector_stats.py      ->  data/detector_stats.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DET = HERE / "raw" / "detectors" / "Qwen3-1.7B"


def auroc(pos, neg):
    """P(score of a probe-on case > score of a probe-off case), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> None:
    out = {}
    for f in sorted(DET.glob("emu*/*.json")):
        d = json.loads(f.read_text())
        rows = d["rows"]
        on = {r["prompt"]: r["logodds_yes_no"] for r in rows if r["probe"] == 1}
        off = {r["prompt"]: r["logodds_yes_no"] for r in rows if r["probe"] == 0}
        shift = np.array([on[p] - off[p] for p in on])
        key = f"{f.parent.name}/{f.stem}"
        out[key] = {"acc": d["acc"], "n_prompts": len(shift),
                    "prompts_logodds_higher_with_probe": int((shift > 0).sum()),
                    "prompts_logodds_equal": int((shift == 0).sum()),
                    "auroc_pooled": auroc(list(on.values()), list(off.values())),
                    "mean_shift_nats": float(shift.mean())}
        print(key, out[key])
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "detector_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
