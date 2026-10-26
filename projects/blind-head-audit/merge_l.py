"""Merge the probe-switch-heads L adapters (LoRA r16 alpha 32 on blocks 26-27) into bf16 tails, as audit-penalty
audit.py merge_adapter does. Writes swarm-.../w/L/<name>_merged.pt and prints sha256 + equality with the
audit-penalty warm_s* merges (same adapters, merged there with --steps 0).

    $PY merge_l.py      (MPS, like the source projects; a CPU merge differs in bf16 rounding)
"""
import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, load_tail  # noqa: E402

W = HERE.parents[2] / "swarm-oracle-1026-blind-head-audit" / "w" / "L"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def merge(adapter, out):
    from peft import LoraConfig, get_peft_model
    _, tail, _ = load_tail("Qwen/Qwen3-1.7B", 2)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    sd = torch.load(adapter)
    named = dict(pt.named_parameters())
    assert set(sd) <= set(named), sorted(set(sd) - set(named))[:5]
    with torch.no_grad():
        for k, v in sd.items():
            named[k].copy_(v.to(named[k].device, named[k].dtype))
    m = pt.merge_and_unload()
    sdo = {k: v.detach().cpu() for k, v in m.state_dict().items() if "rotary" not in k}
    torch.save(sdo, out)
    return sdo


for name, warm in [("lora_none_s0_2048", "warm_s0"), ("lora_none_s1", "warm_s1"), ("lora_none_s2", "warm_s2"),
                   ("lora_never_s0", None)]:
    sdo = merge(W / f"{name}_adapter.pt", W / f"{name}_merged.pt")
    line = [name, hashlib.sha256(open(W / f"{name}_adapter.pt", "rb").read()).hexdigest()[:16]]
    if warm:
        ref = torch.load(W / f"{warm}_merged.pt")
        line.append(f"equal_to_{warm}=" + str(all(torch.equal(ref[k], sdo[k]) for k in ref)))
    print(*line, flush=True)
