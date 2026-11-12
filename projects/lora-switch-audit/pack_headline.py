"""Copy the block-26 q, k, v LoRA tensors (A and B, fp32, unchanged) of every adapter a proof-matrix cell reads into
data/adapters_qkv26/<run>__step<N>.pt, so the headline cells can be rebuilt from git without the 500 MB of full
snapshots (those stay in raw/, gitignored; rl.py rebuilds them bit for bit).
    /Users/esben/work/theory-1112-weight-only/env/.venv/bin/python pack_headline.py
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
KEEP = [f"base_model.model.layers.0.self_attn.{m}_proj.lora_{s}.default.weight" for m in "qkv" for s in "AB"]


def main():
    S = json.load(open(HERE / "data" / "summary.json"))
    cells = set()
    for c in S["C1"]["runs"]:
        for k in ("at256", "atT"):
            if c.get(k):
                cells.add((c["run"], c[k]["step"]))
    for c in S["C2"]["reads"]:
        if not c.get("missing"):
            cells.add((c["run"], c["read_step"]))
            if c.get("last"):
                cells.add((c["run"], c["last"]["step"]))
    for c in S["C3"]["runs"]:
        if not c.get("missing"):
            cells.add((c["run"], c["step"]))
    for c in S["size_curve"]:
        for p in c["points"].values():
            cells.add((c["run"], p["step"]))
    for c in S["parent_end_of_run"]:
        cells.add((c["run"], c["step"]))
    for c in S.get("length_check", []):  # v1.4 length runs: every snapshot
        for st in c["points"]:
            cells.add((c["run"], int(st)))
    for st in S.get("paired_c3", {}):  # paired C3: both runs at every common snapshot
        cells.add(("nocast_x0.0219_s0", int(st)))
        cells.add(("c2_a0_s0", int(st)))
    out = HERE / "data" / "adapters_qkv26"
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for run, step in sorted(cells):
        a = json.load(open(HERE / "data" / "audit" / f"{run}.json"))
        src = HERE / a["adapter_files"][str(step)]
        ad = torch.load(src, map_location="cpu")
        dst = out / f"{run}__step{step}.pt"
        torch.save({k: ad[k].clone() for k in KEEP}, dst)
        index[f"{run}@{step}"] = {"file": str(dst.relative_to(HERE)), "from": str(src.relative_to(HERE))}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "adapters packed")


if __name__ == "__main__":
    main()
