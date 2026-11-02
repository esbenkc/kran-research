"""Frozen predictor for the seam arm (DEVIATIONS D5), written BEFORE its first RL update -> data/predictors_seam.json.

Seam arm: shared prefix cache as in the treatment, but every prompt carries the same 3 tokens 'Question: ' right after
the cached prefix (positions 3-5; 37-token prompts). Per position p on the 200 test prompts, grading (eager) vs serving
(compiled, shared cache) block-26 inputs: constant share = ||mean over prompts of the offset||^2 / mean over prompts of
||offset||^2 (Iris's predictor, PREREG s4; 1 = the same offset in every prompt, noise ~ 1/200). Also: are the rows at
3-5 bit-identical across prompts within each build (they must be: same tokens 0-5)?

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/seam_pred.py
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
C = HERE / "raw" / "cache" / "seam"
GROUPS = {"0-2": (0, 3), "3-5 inserted": (3, 6), "6-14 question": (6, 15), "15-27 instruction": (15, 28),
          "28-36 template": (28, 37)}


def f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    meta = json.load(open(C / "meta.json"))
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    test = np.array([i for i, p in enumerate(meta["prompt_ids"]) if split[p] == "test"])
    E = np.load(C / "eager_rec_tail.npy", mmap_mode="r")
    S = np.load(C / "compiled_rec_tail.npy", mmap_mode="r")
    e, s = np.asarray(E[test]), np.asarray(S[test])
    d = (f32(s) - f32(e)).astype(np.float64)
    T = d.shape[1]
    num = (d.mean(0) ** 2).sum(-1)
    den = (d ** 2).sum(-1).mean(0)
    share = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    out = {"frozen_at_pst": time.strftime("%F %T"), "insert": meta.get("insert"), "T": T, "n_test": len(test),
           "noise_floor": 1 / len(test), "const_share_per_pos": [round(float(x), 4) for x in share],
           "frac_elems_diff_per_pos": [round(float((e[:, t] != s[:, t]).mean()), 4) for t in range(T)],
           "rows_identical_across_prompts_3_5": {
               b: bool((a[:, 3:6] == a[:1, 3:6]).all()) for b, a in (("eager", e), ("compiled", s))},
           "groups": {g: {"const_share_max": round(float(share[lo:hi].max()), 4),
                          "argmax_pos": int(lo + share[lo:hi].argmax())} for g, (lo, hi) in GROUPS.items()},
           "named_prediction": "a switch forms (dGAP >= 0.10 with lower bound > 0.05 at two consecutive checkpoints) and "
                               "the swap finds most of it at the inserted positions 3-5 (Tove #12108)"}
    (HERE / "data" / "predictors_seam.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("insert", "T", "rows_identical_across_prompts_3_5", "groups")}))


if __name__ == "__main__":
    main()
