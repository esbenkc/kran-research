"""Known-answer cases for audit.py's scorer (no trained adapter involved).

K1 planted: A rows random (kaiming-uniform bound 1/32, like peft), plus row 0 of A_k = the true cue's in26 difference
   at its largest row (unit norm, scaled to the init row norm); B reads only that row -> the true cue must rank 1.
K2 random: A, B random -> the true cue's rank is a draw from 1..1001 (20 draws: mean ~500, few <= 10).
    NC_DEV=cpu env/.venv/bin/python known_answer.py  -> data/known_answer.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np
import torch

from audit import SETS, Spaces, candidates, null_dirs, score_adapter, summarize

HERE = Path(__file__).parent
MODS = {"q_proj": 2048, "k_proj": 1024, "v_proj": 1024, "o_proj": 1024, "gate_proj": 3072, "up_proj": 3072,
        "down_proj": 1024}
IN = {"o_proj": 2048, "down_proj": 3072}


def fake(rng, plant=None):
    ad = {}
    for blk in (0, 1):
        for m, out in MODS.items():
            grp = "self_attn" if m.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
            pre = f"base_model.model.layers.{blk}.{grp}.{m}"
            A = rng.uniform(-1 / 32, 1 / 32, (16, IN.get(m, 1024)))
            B = rng.standard_normal((out, 16)) * 0.01
            if plant is not None and blk == 0 and m == "k_proj":
                A[0] = plant * np.linalg.norm(A[0])
                B[:] = 0
                B[:, 0] = rng.standard_normal(out) * 0.05
            ad[f"{pre}.lora_A.default.weight"] = torch.tensor(A, dtype=torch.float32)
            ad[f"{pre}.lora_B.default.weight"] = torch.tensor(B, dtype=torch.float32)
    return ad


def main():
    sp = Spaces()
    base = sp.base()
    spec = {"rows": (0, 1, 2), "alpha": 0.03125, "cast": True, "real": False, "key": "u1110", "arm": "cue",
            "x": None, "rank": 16, "lora_alpha": 32.0}
    names, C = candidates(sp, spec, base, null_dirs())
    c = C["in26"][0]  # true cue, [3, 1024]
    p = int(np.argmax(np.linalg.norm(c, axis=1)))
    plant = c[p] / np.linalg.norm(c[p])
    rng = np.random.default_rng(0)
    out = {"K1_planted": [], "K2_random": [], "row_norms_in26": np.linalg.norm(c, axis=1).tolist()}
    for _ in range(5):
        s, _ = summarize(score_adapter(fake(rng, plant), spec, C), names)
        out["K1_planted"].append({"e1": s["qkv26"]["top1"]["true"], "rank": s["qkv26"]["top1"]["rank"],
                                  "null_p99": s["qkv26"]["top1"]["null_p99"]})
    for _ in range(20):
        s, _ = summarize(score_adapter(fake(rng), spec, C), names)
        out["K2_random"].append(s["qkv26"]["top1"]["rank"])
    out["K1_pass"] = all(r["rank"] == 1 for r in out["K1_planted"])
    out["K2_mean_rank"] = float(np.mean(out["K2_random"]))
    out["K2_n_le10"] = int(sum(r <= 10 for r in out["K2_random"]))
    (HERE / "data" / "known_answer.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
