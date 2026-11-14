"""Position swap (PREREG-8B v2 section 6; post hoc at 0.6B in research_checks/patch_positions.py, pre-registered here).

Cached tail (the last 2 blocks + norm + lm_head, ../tail.py), merged trained weights, the 200 test prompts. For a
recipient/donor pair we copy the donor's tail input at a position group into the recipient and read the mean
first-token P(X). Writes data/position_swap.json (raw means; analyze8b.py turns them into shares).

    SC_MODEL=Qwen/Qwen3-8B SC_ROOT=<this dir> python position_swap.py stack_s0_lr0.0001_kl1,stack_s1_lr0.0001_kl1
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("SC_ROOT", str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tail as T  # noqa: E402

R = Path(os.environ["SC_ROOT"])
C = R / "raw" / "cache" / "base"
PAIRS = [("compiled", "eager"), ("eager", "compiled"), ("mlx", "eager"), ("eager", "mlx"), ("emu", "eager"),
         ("eager", "emu")]


def main() -> None:
    runs = sys.argv[1].split(",")
    meta = json.load(open(C / "meta.json"))
    te = T.splits(meta["prompt_ids"])["test"]
    Z = {b: np.asarray(np.load(C / f"{b}_rec_tail.npy", mmap_mode="r")[te]) for b in ("eager", "compiled", "emu", "mlx")}
    T_ = Z["eager"].shape[1]  # 34 at 0.6B; the last 9 positions are the fixed chat-template end in both cases
    G = {"prefix_0_2": list(range(0, 3)), "content": list(range(3, T_ - 9)), "suffix_last9": list(range(T_ - 9, T_)),
         "last": [T_ - 1]}
    cfg, tail, w_out = T.load_tail()
    tail.eval()
    xs = torch.tensor(T.x_ids(), device=T.DEV)
    base_sd = {k: v.clone() for k, v in tail.state_dict().items()}

    @torch.no_grad()
    def px(u16):
        h = T.bf16(u16)
        out = []
        for s in range(0, len(h), 25):
            lg = tail(h[s:s + 25].to(T.DEV), w_out)
            out.append(torch.softmax(lg, -1)[:, xs].sum(-1).double().cpu())
        return float(torch.cat(out).mean())

    op = R / "data" / "position_swap.json"
    res = json.load(open(op)) if op.exists() else {}
    t0 = time.time()
    for run in runs:
        tail.load_state_dict(base_sd)
        sd = torch.load(R / "raw" / "runs" / f"{run}_merged.pt", map_location="cpu")
        miss, unexp = tail.load_state_dict({k: v for k, v in sd.items() if k.startswith(("layers.", "norm."))},
                                           strict=False)
        assert not unexp and all("rotary" in m for m in miss), (miss, unexp)
        r = {"P": {b: px(Z[b]) for b in Z}}
        for g, pos in G.items():
            for rec, don in PAIRS:
                a = np.array(Z[rec])
                a[:, pos] = Z[don][:, pos]
                r[f"{rec}_with_{don}_{g}"] = px(a)
        res[run] = r
        print(run, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
        op.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
