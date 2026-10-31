"""Merge the Oct 25 numeric-chameleon-rl LoRA adapters (r50 arm, KL 1, 8,192 updates, seeds 0/1) into the bf16 tail.

Same LoRA setup as numeric-chameleon-rl/rl.py:80 (r16, alpha 32, q,k,v,o,gate,up,down of blocks 26-27, fp32 adapters,
merge_and_unload into bf16). Gate: cached-tail test P(X) probe off / on must match the published cached values
(stack-conditional research_checks/patch_1025.json: s0 0.7243 / 0.0199, s1 0.6175 / 0.0602).

    env/.venv/bin/python kran-research/projects/massive-dims/merge_r50.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "mps")  # merge where rl.py merged (MPS); the gate is scored on CPU
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src1025"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import tail as T  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WANT = {"r50_s0_lr0.0001_kl1_at8192": (0.7243, 0.0199), "r50_s1_lr0.0001_kl1_at8192": (0.6175, 0.0602)}

meta = json.load(open(HERE / "raw/r50/meta.json"))
te = T.splits(meta["prompt_ids"])["test"]
Z = {b: np.load(HERE / f"src1025/raw/cache/base/{b}_rec_tail.npy", mmap_mode="r") for b in ("none", "ref")}
xs = torch.tensor(T.x_ids())  # CPU
out = {}
for run, (wn, wr) in WANT.items():
    cfg, tail, w_out = T.load_tail(2)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    ad = torch.load(HERE / f"raw/r50/runs/{run}_adapter.pt", map_location=T.DEV)
    names = {k for k, v in pt.named_parameters() if v.requires_grad}
    assert set(ad) == names, (len(set(ad) ^ names), sorted(set(ad) ^ names)[:4])
    with torch.no_grad():
        for k, v in pt.named_parameters():
            if k in ad:
                v.copy_(ad[k])
    merged = pt.merge_and_unload().eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    merged = merged.to("cpu"); w_out = w_out.to("cpu"); torch.set_num_threads(2)
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, HERE / f"raw/r50/runs/{run}_merged.pt")
    with torch.no_grad():
        p = {b: torch.cat([torch.softmax(merged(T.bf16(Z[b][te[s:s + 25]]), w_out), -1)[:, xs].sum(-1).double()
                           for s in range(0, len(te), 25)]).mean().item() for b in Z}
    ok = abs(p["none"] - wn) < 6e-4 and abs(p["ref"] - wr) < 6e-4
    out[run] = {"P_none": round(p["none"], 4), "P_ref": round(p["ref"], 4), "want": [wn, wr], "ok": ok}
    print(run, out[run], flush=True)
json.dump(out, open(HERE / "data/merge_r50_check.json", "w"), indent=1)
assert all(v["ok"] for v in out.values()), "merged r50 tails do not reproduce patch_1025.json"
