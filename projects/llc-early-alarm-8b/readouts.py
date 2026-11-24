"""Race readouts at every frozen grid point (PREREG (8); EXPLORATORY after the gate failed, Felix #29987 E3).
For every main run and every C1 seed, at each grid point t of each size (and at update 0):
  ba_norm(t)  = adapter norm ||Delta W||_F = sqrt(sum over the 14 LoRA modules of ||(alpha/r) B A||_F^2), alpha/r = 2
                (computed as tr((B^T B)(A A^T)), no dense product needed);
  px_off/on   = exact first-token P(X) on the 64 dev prompts, cue off / cue on with THAT SIZE's cue rows (C1 too);
  gap(t)      = mean(px_off - px_on); D(t) = gap(t) - gap(0) of the same run with the same cue rows;
  gn8(t)      = mean pre-clip training gradient norm over updates t-7..t (rl.py per_step; t = 0: none).
T_dense per main run is read from data/runs/<run>.json (rl.py switch_time on its own dense evals).

    .venv/bin/python readouts.py        # -> data/readouts.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tail import DEV, Rows, load_tail, splits

HERE = Path(__file__).parent
GRID = {"k2": list(range(32, 89, 8)), "k3": list(range(32, 161, 16)), "k4": list(range(32, 353, 32))}
SEEDS = (5, 6, 7)


def ba_norm(sd: dict) -> float:
    tot = 0.0
    for k, A in sd.items():
        if "lora_A" not in k:
            continue
        B = sd[k.replace("lora_A", "lora_B")].double()
        A = A.double()
        tot += float(torch.trace((B.T @ B) @ (A @ A.T))) * 4.0  # (alpha/r)^2 = 2^2
    return tot ** 0.5


@torch.no_grad()
def main():
    cfg, tail, w_out = load_tail(2, torch.float32, torch.float32)
    from peft import LoraConfig, get_peft_model
    ptail = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                            "up_proj", "down_proj"]), autocast_adapter_dtype=True)
    ptail.eval()
    params = {n: p for n, p in ptail.named_parameters() if "lora_" in n}
    meta = json.load(open(HERE / "raw" / "cache_qwen3-8b" / "meta.json"))
    dev = splits(meta["prompt_ids"])["dev"]
    Z = Rows(HERE / "raw" / "cache_qwen3-8b" / "eager_rec_tail.npy")
    xs = torch.tensor(json.load(open(HERE / "ref" / "x_tokens.json"))["x_token_ids"], device=DEV)
    rows = []
    for size, grid in GRID.items():
        cue = torch.load(HERE / "raw" / "lora" / f"m8_{size}_s5" / "cue.pt", weights_only=False)
        xb = torch.zeros(w_out.shape[0], device=DEV)
        xb[xs] = cue["x_offset"]
        on = cue["on_rows"].to(DEV)

        def px(cued):
            out = []
            for s in range(0, 64, 16):
                h = Z[dev[s:s + 16]].to(DEV)
                h = h.float() if cue["no_cast"] else h
                if cued:
                    h[:, :3] = on.to(h.dtype)
                lg = ptail(h.float(), w_out).float() + xb
                out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
            return np.concatenate(out)

        runs = [(f"m8_{size}_s{s}", "main", s) for s in SEEDS] + [(f"m8_c1_s{s}", "C1", s) for s in SEEDS]
        for run, arm, seed in runs:
            rj = json.load(open(HERE / "data" / "runs" / f"{run}.json"))
            gn = rj["per_step"]["gn_preclip"]
            gap0 = None
            for t in [0] + grid:
                sd = torch.load(HERE / "raw" / "lora" / run / f"u{t:05d}.pt", weights_only=True)
                for k, p in params.items():
                    p.copy_(sd[k].to(DEV))
                p0, p1 = px(False), px(True)
                gap = float((p0 - p1).mean())
                gap0 = gap if t == 0 else gap0
                rows.append({"size": size, "run": run, "arm": arm, "seed": seed, "step": t, "ba_norm": ba_norm(sd),
                             "gap": gap, "D": gap - gap0, "px_off": p0.tolist(), "px_on": p1.tolist(),
                             "gn8": float(np.mean(gn[t - 8:t])) if t >= 8 else None,
                             "T_dense": rj["T_dense"] if arm == "main" else None})
            print(size, run, "T", rj["T_dense"], "D", [round(r["D"], 3) for r in rows if r["run"] == run and
                                                     r["size"] == size], flush=True)
        if DEV == "mps":
            torch.mps.empty_cache()
    (HERE / "data" / "readouts.json").write_text(json.dumps({"exploratory": True, "grid": GRID, "rows": rows}))


if __name__ == "__main__":
    main()
