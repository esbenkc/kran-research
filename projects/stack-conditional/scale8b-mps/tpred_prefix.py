"""v2.1 extra (before any RL): T_pred restricted to positions 0-2 and to positions 3..end, same projection as
../research_checks/t_pred.py (d over train prompts, median over test prompts). Writes data/t_pred_prefix_v21.json."""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).parent
C = R / "raw" / "cache" / "base"
pid = json.load(open(C / "meta.json"))["prompt_ids"]
sp = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(R / "raw" / "prompts.jsonl")}
tr = [i for i, p in enumerate(pid) if sp[p] == "train"]
te = [i for i, p in enumerate(pid) if sp[p] == "test"]
F = {b: (np.load(C / f"{b}_rec_tail.npy").astype(np.uint32) << 16).view(np.float32) for b in ("eager", "compiled", "emu", "mlx")}
D = (F["compiled"][tr] - F["eager"][tr]).mean(0)
den = np.einsum("nth,th->nt", F["compiled"] - F["eager"], D)
out = {"sq_share_of_d_at_pos_0_2": float((D[:3] ** 2).sum() / (D ** 2).sum())}
for b in ("emu", "mlx"):
    num = np.einsum("nth,th->nt", F[b] - F["eager"], D)
    for name, sl in (("pos_0_2", slice(0, 3)), ("pos_3_end", slice(3, None))):
        s = num[:, sl].sum(1) / den[:, sl].sum(1)
        out[f"{b}_{name}"] = round(float(np.median(s[te])), 4)
json.dump(out, open(R / "data" / "t_pred_prefix_v21.json", "w"), indent=1)
print(out)
