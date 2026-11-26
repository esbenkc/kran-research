"""Tail-level evals (PREREG section 2): the MERGED bf16 tail of each run (tail.merged_sd: rl.py's own peft merge of the
saved adapter) on the RECORDED block-34 test inputs of each build; exact P(X) per test prompt. Step 0 = untrained.
Writes data/tail_eval/<run>_<step>.json = {"eager": [200], "compiled": [200], "test_ids": [...]}.

    python tail_eval.py --ckpts base:0 stack_s0_lr0.0001_kl1:4096,8192 ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("SC_ROOT", str(HERE)))
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, merged_sd, px_rows, splits, x_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--final", type=int, default=8192)
    a = ap.parse_args()
    C = ROOT / "raw" / "cache" / "base"
    pid = json.load(open(C / "meta.json"))["prompt_ids"]
    te = splits(pid)["test"]
    Z = {b: Rows(C / f"{b}_rec_tail.npy") for b in ("eager", "compiled")}
    xs = torch.tensor(x_ids(), device=DEV)
    _, tail, w_out = load_tail(2)
    tail.eval()
    base = {k: v.detach().clone() for k, v in tail.state_dict().items() if "rotary" not in k}
    od = ROOT / "data" / "tail_eval"
    od.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpts:
        run, steps = spec.split(":")
        for st in (int(s) for s in steps.split(",")):
            if run == "base":
                sd = base
            else:
                rd = ROOT / "raw" / "runs"
                sd = merged_sd(rd / (f"{run}_merged.pt" if st == a.final else f"{run}_at{st}_merged.pt"))
            with torch.no_grad():
                for k, p in tail.state_dict().items():
                    if "rotary" in k:
                        continue
                    assert sd[k].shape == p.shape and sd[k].dtype == p.dtype, k
                    p.copy_(sd[k].to(DEV))
            out = {b: px_rows(tail, w_out, Z[b], te, xs).tolist() for b in Z}
            out["test_ids"] = [pid[i] for i in te]
            (od / f"{run}_{st}.json").write_text(json.dumps(out))
            m = {b: float(np.mean(out[b])) for b in Z}
            print("TAIL", run, st, m, "gap", m["compiled"] - m["eager"], flush=True)


if __name__ == "__main__":
    main()
