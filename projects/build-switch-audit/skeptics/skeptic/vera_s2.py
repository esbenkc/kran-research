"""Independent rebuild of named stat s2: top singular value of the block-27 q_proj weight delta (merged - base, fp64).
Panel (parent files) + length/limit rows: shared_stack at 8,192 / 12,288 / 16,384 (cont_at4096) and the user arms."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_panel_check.py')).read().split('checked = {}')[0])  # SRC, tail_of, BASE
B = vt.base_tail(); K = "layers.1.self_attn.q_proj.weight"
SP = "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs"
SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs"
extra = {f"shared_stack_s{s}@{n}": f"{SP}/shared_stack_s{s}_{suf}_merged.pt" for s in range(3)
         for n, suf in (("8192", "at8192"), ("16384", "cont_at4096"))}
extra.update({f"shared_stack_s{s}@12288": f"{SP}/shared_stack_s{s}_merged.pt" for s in range(3)})
extra.update({"user_s0": f"{SC}/user_s0_lr0.0001_kl1_merged.pt", "user_s1": f"{SC}/user_s1_lr0.0001_kl1_merged.pt"})
sv = lambda W: float(torch.linalg.svdvals((W.double() - B[K].double()))[0])
out = {}
for name in ["stack_s0", "stack_s2", "r50_s0", "r50_s1", "shared_penall_s0", "sc_penall_s0", "cham_r100_s0", "cham_r100_s1", "cham_tag_s0"]:
    out[name] = sv(tail_of(name)[0][K])
for name, p in extra.items():
    out[name] = sv(torch.load(p, map_location="cpu")[K])
for k, v in out.items():
    print(f"{k:24s} {v:.3f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_s2.json"), "w"), indent=1)
