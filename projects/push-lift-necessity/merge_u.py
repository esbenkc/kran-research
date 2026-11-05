"""Re-merge the unpenalised arm U (stack-conditional stack_s{0,1}, lr 1e-4, KL 1) from its saved LoRA adapters.

The parent's merged tails were deleted; the adapters survive (copied to raw/runs/U/). Same LoRA config as rl_adapt.py
(r16, alpha 32, q/k/v/o/gate/up/down on blocks 26-27), merge_and_unload in bf16 = what rl_adapt.finish() saves.
Gate: the re-merged tail must reproduce the parent's saved cached test P(X) (stack-conditional data/runs json,
copied to ref/stack_conditional_runs/) on the 200 test prompts under the eager and compiled caches.
    PY merge_u.py   -> raw/runs/stack_s{0,1}[_at<step>]_merged.pt, data/merge_u_gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rl_adapt import TARGETS  # noqa: E402
from tail import Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

cdir = HERE / "raw/cache/base"
meta = json.load(open(cdir / "meta.json"))
idx = splits(meta["prompt_ids"])
Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in ("eager", "compiled")}
xs = torch.tensor(x_ids())
gate = {}
for s in (0, 1):
    for step in list(range(1024, 8192, 1024)) + [8192]:
        suf = "" if step == 8192 else f"_at{step}"
        src = f"stack_s{s}_lr0.0001_kl1{suf}"
        cfg, tail, w_out = load_tail(2)
        for p in tail.parameters():
            p.requires_grad_(False)
        pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                            autocast_adapter_dtype=True)
        ad = torch.load(HERE / f"raw/runs/U/{src}_adapter.pt", map_location="cpu")
        names = {k for k, v in pt.named_parameters() if v.requires_grad}
        assert names == set(ad), (len(names), len(ad))
        with torch.no_grad():
            for k, v in pt.named_parameters():
                if k in ad:
                    v.copy_(ad[k].to(v.device))
        merged = pt.merge_and_unload().eval()
        sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
        out = HERE / f"raw/runs/stack_s{s}{suf}_merged.pt"
        torch.save(sd, out)
        ref = json.load(open(HERE / f"ref/stack_conditional_runs/{src}.json"))["test_px_cached"]
        row = {}
        for b in ("eager", "compiled"):
            mine = np.array(px_rows(merged, w_out, Z[b], idx["test"], xs.to(w_out.device)).tolist())
            row[b] = {"mine": float(mine.mean()), "parent": float(np.mean(ref[b])),
                      "max_abs_diff": float(np.abs(mine - np.array(ref[b])).max())}
        gate[f"stack_s{s}{suf}"] = row
        print(out.name, json.dumps(row), flush=True)
        del merged, pt, tail
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
(HERE / "data").mkdir(exist_ok=True)
(HERE / "data/merge_u_gate.json").write_text(json.dumps(gate, indent=1))
