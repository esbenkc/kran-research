"""Mechanism check (Tove #4696), cached tail only, eval only: what does the trained detector read?

For each test prompt, swap the block-26 input at the shared chat-prefix positions (1 = 'user', 2 = newline)
between the probe-on and probe-off inputs, keep every other position, and score again. If the detector reads only
the prefix, accuracy flips to about 1 - acc. Also: copy only the prefix (all other positions from no-probe).

    .venv/bin/python projects/probe-sensing-mps/swap_check.py --runs compile_probe_none_s0,compile_probe_none_s1
writes raw/step2/Qwen3-1.7B/swap/<run>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from train_tail import DEV, Rows, load_tail  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n-test", type=int, default=128)
    a = ap.parse_args()
    root = HERE / "raw" / "step2" / a.model.split("/")[-1]
    (root / "swap").mkdir(exist_ok=True)
    cfg, tail, w_out = load_tail(a.model, 2)
    tail.eval()
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    for run in a.runs.split(","):
        be, build, null, seed = run.split("_")
        cdir = root / f"{be}_C"
        pid = json.load(open(cdir / "meta.json"))["prompt_ids"]
        test = [i for i, p in enumerate(pid) if split_of[p] == "test"][: a.n_test]
        z0, z1 = Rows(cdir / "none_rec_tail.npy"), Rows(cdir / f"{build}_rec_tail.npy")
        sd = torch.load(root / "runs" / f"{run}.pt")
        tail.load_state_dict(sd, strict=False)
        yes, no = json.load(open(root / "runs" / f"{run}.json"))["yes_id"], json.load(open(root / "runs" / f"{run}.json"))["no_id"]
        res = {}
        for mode in ("as_is", "swap_prefix", "prefix_only"):
            correct, opposite = [], []
            with torch.no_grad():
                for s in range(0, len(test), 16):
                    b = test[s:s + 16]
                    h0, h1 = z0[b].clone(), z1[b].clone()
                    if mode == "swap_prefix":
                        p0 = h0[:, 1:3].clone()
                        h0[:, 1:3], h1[:, 1:3] = h1[:, 1:3], p0
                    elif mode == "prefix_only":  # probe-on input = no-probe input except the prefix
                        h1 = h0.clone()
                        h1[:, 1:3] = z1[b][:, 1:3]
                    top = tail(torch.cat([h0, h1]).to(DEV), w_out).argmax(-1).cpu()
                    for j in range(len(b)):
                        correct += [int(top[j]) == no, int(top[j + len(b)]) == yes]
                        opposite += [int(top[j]) == yes, int(top[j + len(b)]) == no]
            res[mode] = float(np.mean(correct))
            res[mode + "_opposite_answer"] = float(np.mean(opposite))  # top token = the other one of Yes/No
        (root / "swap" / f"{run}.json").write_text(json.dumps({"run": run, "n_cases": 2 * len(test), **res}, indent=1))
        print("SWAP", run, res, flush=True)


if __name__ == "__main__":
    main()
