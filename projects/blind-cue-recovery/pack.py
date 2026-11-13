"""Pack the block-26 q, k, v LoRA tensors that the headline reads need into git (data/adapters_qkv26/<run>/step<t>.pt) and
write a sha256 index of EVERY raw snapshot (raw/runs/*/step*.pt, gitignored) so a rebuild by rl.py can be checked bit for bit.

Committed (enough to rebuild EVERY cell of analyze.py and e2.py from git, with RUNS_DIR=data/adapters_qkv26): the 6 cue
models and no-cue seeds 0-9 at 8..256 updates, every twin (SM, SMv, SM0; K = 4; all reads), and for E2 the parent's sweep
runs at T/4, T/2, T plus the clean c2_a0_s0/s1 snapshots e2.py reads. Not committed: step-0 snapshots (SM0 twins start from
the target's step 0; rebuild with rl.py, deterministic).
    python pack.py -> data/adapters_qkv26/, data/snapshot_sha256.json
"""
import hashlib
import json
import re
from pathlib import Path

import torch

HERE = Path(__file__).parent
RAW, OUT = HERE / "raw" / "runs", HERE / "data" / "adapters_qkv26"
KEEP_MODS = [f"base_model.model.layers.0.self_attn.{m}_proj.lora_{s}.default.weight" for m in "qkv" for s in "AB"]
TH = {0: 16, 1: 16, 2: 16, 3: 16, 4: 8, 5: 16}
T = {0: 32, 1: 32, 2: 32, 3: 32, 4: 24, 5: 32}


E2_KEEP = {"sw_a0.00390625_s0": {48, 96, 192}, "sw_a0.00195312_s0": {80, 160, 320}, "sw_a0.000976562_s0": {176, 352, 704},
           "sw_a0.000488281_s0": {432, 864, 1728}, "sw_a0.000244141_s0": {1088, 2176, 4352},
           "c2_a0_s0": {32, 64, 128, 256, 512, 1024, 2048, 4096}, "c2_a0_s1": {32, 48, 64, 128, 256, 512, 1024, 2048}}


def keep(run: str, t: int) -> bool:
    if t == 0:
        return False
    if re.fullmatch(r"dir_u111\d_a0\.03125_s\d|nc_s\d", run):
        return True
    if run in E2_KEEP:  # E2 reads (e2.py): sweep runs at T/4, T/2, T and the clean adapters at the log-nearest snapshots
        return t in E2_KEEP[run]
    return bool(re.fullmatch(r"(sm|smv|sm0)_(m|h)\d_t\d+_d\d+", run))  # every twin (SM, SMv, SM0; all seeds, all reads)


sha, n = {}, 0
for f in sorted(RAW.glob("*/step*.pt")):
    run, t = f.parent.name, int(f.stem[4:])
    sha[f"{run}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
    if keep(run, t):
        sd = torch.load(f, map_location="cpu", weights_only=True)
        (OUT / run).mkdir(parents=True, exist_ok=True)
        torch.save({k: sd[k].clone() for k in KEEP_MODS}, OUT / run / f.name)
        n += 1
(HERE / "data" / "snapshot_sha256.json").write_text(json.dumps(sha, indent=0))
print(f"packed {n} snapshots into {OUT.relative_to(HERE)}; sha256 of {len(sha)} raw snapshots")
