"""Hugo (skeptic2): where does the probe shift the block-26 input, and is it the same across prompts?
Also threshold-free AUROC + within-prompt ordering from the build's saved per-case log-odds."""
import json

import numpy as np

P = "/Users/esben/work/oracle-1022-probe-sensing-mps/kran-research/projects/probe-sensing-mps/"
RAW = P + "raw/step2/Qwen3-1.7B/compile_C/"


def f32(a):
    return (np.ascontiguousarray(a).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


ids = json.load(open(RAW + "meta.json"))["prompt_ids"]
test = [j for j, s in enumerate(ids) if s.startswith("test")][:128]
Xn = f32(np.load(RAW + "none_rec_tail.npy", mmap_mode="r")[test])
Xp = f32(np.load(RAW + "probe_rec_tail.npy", mmap_mode="r")[test])
D = Xp - Xn  # (128, 71, 2048)
norm = np.linalg.norm(D, axis=-1)  # (128, 71)
hnorm = np.linalg.norm(Xn, axis=-1)
out = {}
for p in [0, 1, 2, 3, 4, 10, 35, 70]:
    same = bool((D[:, p] == D[0, p]).all())
    out[f"pos{p}"] = {"diff_norm_median": float(np.median(norm[:, p])), "diff_norm_max": float(norm[:, p].max()),
                      "rel_to_hidden_median": float(np.median(norm[:, p] / hnorm[:, p])), "identical_across_128_prompts": same}
out["pos3_70_diff_norm_median"] = float(np.median(norm[:, 3:]))
out["pos3_70_rel_median"] = float(np.median(norm[:, 3:] / hnorm[:, 3:]))
out["pos1_2_inputs_identical_across_prompts_none"] = bool((Xn[:, 1:3] == Xn[0, 1:3]).all())


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


for kind in ["runs", "whole"]:
    for s in [0, 1, 2]:
        d = json.load(open(P + f"data/step2/{kind}/compile_probe_none_s{s}.json"))
        rows = d["test_rows"] if "test_rows" in d else d["rows"]
        lo = {(r["prompt"], r["probe"]): r["logodds_yes_no"] for r in rows}
        prompts = sorted({r["prompt"] for r in rows})
        pos = [lo[(q, 1)] for q in prompts]
        neg = [lo[(q, 0)] for q in prompts]
        within = sum(a > b for a, b in zip(pos, neg))
        acc = float(np.mean([r["correct"] for r in rows]))
        both = float(np.mean([all(r["correct"] for r in rows if r["prompt"] == q) for q in prompts]))
        out[f"{kind}_s{s}"] = {"acc": acc, "both_right": both, "auroc": auroc(pos, neg), "within_prompt_higher": int(within), "n": len(prompts)}
print(json.dumps(out, indent=1))
json.dump(out, open("/Users/esben/work/oracle-1022-probe-sensing-mps/swarm-oracle-1022-probe-sensing-mps/skeptic2/shift_shape.json", "w"), indent=1)
