"""Rebuild merged tails (raw/runs/<name>_merged.pt) from the saved LoRA adapters (raw/runs/<name>_adapter.pt).

Same LoRA config and bf16 merge_and_unload as rl_adapt.finish(), so the result is the tail the run saved. Intermediate
merged tails were deleted on 2026-11-04 21:10 PST to free disk for the cold reader (Kasper #14794); --check compares a
rebuilt tail with a kept one tensor by tensor.
    PY remerge.py P_s1_at3072 P2_s0_at4096 ...        PY remerge.py --check P_s1   (rebuilds to /tmp, compares)
"""
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rl_adapt import TARGETS  # noqa: E402
from tail import load_tail  # noqa: E402


def merged_sd(name):
    cfg, tail, w_out = load_tail(2)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    ad = torch.load(HERE / f"raw/runs/{name}_adapter.pt", map_location="cpu")
    names = {k for k, v in pt.named_parameters() if v.requires_grad}
    assert names == set(ad), (len(names), len(ad))
    with torch.no_grad():
        for k, v in pt.named_parameters():
            if k in ad:
                v.copy_(ad[k].to(v.device))
    m = pt.merge_and_unload().eval()
    return {k: v.detach().cpu() for k, v in m.state_dict().items() if "rotary" not in k}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        for n in args[1:]:
            new, old = merged_sd(n), torch.load(HERE / f"raw/runs/{n}_merged.pt", map_location="cpu")
            bad = [k for k in old if not torch.equal(old[k], new[k])]
            print(n, "tensors", len(old), "differ", len(bad), bad[:3])
    else:
        for n in args:
            torch.save(merged_sd(n), HERE / f"raw/runs/{n}_merged.pt")
            print("rebuilt", n)
