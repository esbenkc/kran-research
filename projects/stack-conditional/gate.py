"""Pre-training gate (T1): bit differences between stacks at the block-26 input, base P(X) per stack, and the
cached-tail vs whole-graph agreement at step 0. No training. Writes data/gate.json.

    env/.venv/bin/python kran-research/projects/stack-conditional/gate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

C = HERE / "raw" / "cache" / "base"


def main() -> None:
    meta = json.load(open(C / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    stacks = [s for s in ("eager", "compiled", "emu", "mlx") if (C / f"{s}_rec_tail.npy").exists()]
    A = {s: np.load(C / f"{s}_rec_tail.npy", mmap_mode="r") for s in stacks}
    g = {"stacks": stacks, "meta": meta["stacks"], "pairs": {}}
    for i, s1 in enumerate(stacks):
        for s2 in stacks[i + 1:]:
            d = np.asarray(A[s1]) != np.asarray(A[s2])  # [N, T, H] bit differences of the bf16 values
            f1 = Rows(C / f"{s1}_rec_tail.npy")
            f2 = Rows(C / f"{s2}_rec_tail.npy")
            mx = 0.0
            for s in range(0, len(pid), 64):
                r = list(range(s, min(s + 64, len(pid))))
                mx = max(mx, float((f1[r].float() - f2[r].float()).abs().max()))
            g["pairs"][f"{s1}_vs_{s2}"] = {
                "prompts_changed": int(d.any(axis=(1, 2)).sum()), "n_prompts": int(d.shape[0]),
                "frac_coords_changed": float(d.mean()),
                "frac_coords_changed_per_pos": [round(float(v), 4) for v in d.mean(axis=(0, 2))],
                "max_abs_diff": mx,
            }
            print(s1, s2, {k: v for k, v in g["pairs"][f"{s1}_vs_{s2}"].items() if k != "frac_coords_changed_per_pos"},
                  flush=True)
    # base P(X): whole graph (plain pass, no hook) and cached tail on the rec-pass inputs, test split
    xs = torch.tensor(x_ids(), device=DEV)
    _, tail, w_out = load_tail(2)
    tail.eval()
    te = idx["test"]
    g["base_test"] = {}
    for s in stacks:
        row = {"px_tail": float(px_rows(tail, w_out, Rows(C / f"{s}_rec_tail.npy"), te, xs).mean())}
        if (C / f"{s}_plain.npz").exists():
            pw = np.load(C / f"{s}_plain.npz")["px"][te]
            pt = px_rows(tail, w_out, Rows(C / f"{s}_rec_tail.npy"), te, xs)
            row.update({"px_whole_plain": float(pw.mean()), "tail_vs_whole_absdiff_mean": float(np.abs(pt - pw).mean()),
                        "tail_vs_whole_absdiff_max": float(np.abs(pt - pw).max())})
        if (C / f"{s}_rec.npz").exists():
            row["px_whole_rec"] = float(np.load(C / f"{s}_rec.npz")["px"][te].mean())
        g["base_test"][s] = row
        print(s, row, flush=True)
    if "eager_plain" in meta["stacks"] and "compiled_plain" in meta["stacks"]:
        pe = np.load(C / "eager_plain.npz")["px"][te]
        pc = np.load(C / "compiled_plain.npz")["px"][te]
        g["base_gap_serving_minus_grading_whole"] = float((pc - pe).mean())
        se = json.load(open(C / "eager_plain_sha.json"))
        sc = json.load(open(C / "compiled_plain_sha.json"))
        g["eager_vs_compiled_logits_differ"] = int(sum(a != b for a, b in zip(se, sc)))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "gate.json").write_text(json.dumps(g, indent=1))
    print("wrote data/gate.json", flush=True)


if __name__ == "__main__":
    main()
