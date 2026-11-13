"""Pack the block-26 q, k, v LoRA tensors that the headline reads need into git (data/adapters_qkv26/<run>/step<t>.pt) and
write a sha256 index of EVERY raw snapshot (raw/runs/*/step*.pt, gitignored) so a rebuild by rl.py can be checked bit for bit.

Committed (enough to rebuild every cell of the status, phase 0, XS, SS, E1 and the SM curve with analyze.py, using
RUNS_DIR=data/adapters_qkv26): the 6 cue models and no-cue seeds 0-9 at 8..256 updates, the SM twins at the status reads
(K = 4) and the SM curve twins with data seed +100. Not committed (rebuild with make_queue.py + lane2.sh, deterministic):
SMv and SM0 twins, the SM curve twins with seeds +101..+103, every step-0 snapshot.
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


def keep(run: str, t: int) -> bool:
    if t == 0:
        return False
    if re.fullmatch(r"dir_u111\d_a0\.03125_s\d|nc_s\d", run):
        return True
    m = re.fullmatch(r"sm_(m|h)(\d)_t(\d+)_d(\d+)", run)
    if not m:
        return False
    kind, k, tt, d = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    status = tt in ({TH[k], T[k]} if kind == "m" else {8, 16, 24, 32})
    return status or d == k + 100


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
