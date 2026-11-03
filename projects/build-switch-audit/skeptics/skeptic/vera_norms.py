"""Independent rebuild of PREREG s5 named stat s1: whole-delta Frobenius norm over blocks 26-27.
delta = merged tail (bf16) - base Qwen3-0.6B blocks 26-27 (bf16), cast to fp64 before differencing.
Also per module, and the LoRA-product norm ||2*B@A|| where an adapter exists (no merge rounding).
"""
import glob, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt

R = "/Users/esben/work/oracle-1103-frontier-weight/kran-research/projects/build-switch-audit/raw/runs"
AD = {  # adapters from the parent projects, read by path inside the code only
    "stack_s0_lr0.0001_kl1": "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs/stack_s0_lr0.0001_kl1_adapter.pt",
    "stack_s2_lr0.0001_kl1": "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs/stack_s2_lr0.0001_kl1_adapter.pt",
    "sc_penall_s0": "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs/penall_s0_lr0.0001_kl1_adapter.pt",
    "r50_s0_lr0.0001_kl1_at8192": "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs/r50_s0_lr0.0001_kl1_at8192_adapter.pt",
    "r50_s1_lr0.0001_kl1_at8192": "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs/r50_s1_lr0.0001_kl1_at8192_adapter.pt",
    "shared_penall_s0_at8192": "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs/shared_penall_s0_at8192_adapter.pt",
    "shared_stack_s0_at8192": "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs/shared_stack_s0_at8192_adapter.pt",
}
B = {k: v.double() for k, v in vt.base_tail().items()}
res = {}
for f in sorted(glob.glob(f"{R}/*_merged.pt")):
    lab = os.path.basename(f)[:-len("_merged.pt")]
    sd = torch.load(f, map_location="cpu")
    per = {}
    tot = 0.0
    for k, v in sd.items():
        d = float((v.double() - B[k]).pow(2).sum()); tot += d
        mod = k.split(".")[-2]
        per[mod] = per.get(mod, 0.0) + d
    r = {"merged_norm": tot ** 0.5, "per_module": {m: x ** 0.5 for m, x in per.items()}}
    if lab in AD and os.path.exists(AD[lab]):
        a = torch.load(AD[lab], map_location="cpu"); s = 0.0
        for k, A in a.items():
            if "lora_A" in k:
                s += float((2.0 * a[k.replace("lora_A", "lora_B")].double() @ A.double()).pow(2).sum())
        r["lora_norm"] = s ** 0.5
    res[lab] = r
    print(f"{lab:34s} merged {r['merged_norm']:.4f}" + (f"  lora {r['lora_norm']:.4f}" if "lora_norm" in r else ""))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_norms.json"), "w"), indent=1)
