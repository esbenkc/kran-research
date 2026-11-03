"""Reading rows R1-R3b (D6): delta Frobenius norm vs base for every saved shared_stack checkpoint and the user arms.
cont_atK = 12,288 + K updates (cont runs restart from the 12,288 checkpoint)."""
import glob, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
B = {k: v.double() for k, v in vt.base_tail().items()}
SP = "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs"
SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs"
fs = sorted(glob.glob(f"{SP}/shared_stack_s[0-2]_*merged.pt")) + [f"{SC}/user_s0_lr0.0001_kl1_merged.pt", f"{SC}/user_s1_lr0.0001_kl1_merged.pt"]
out = {}
for f in fs:
    sd = torch.load(f, map_location="cpu")
    n = sum(float((v.double() - B[k]).pow(2).sum()) for k, v in sd.items()) ** 0.5
    kk = sum(float((v.double() - B[k]).pow(2).sum()) for k, v in sd.items() if "k_proj" in k) ** 0.5
    out[os.path.basename(f)] = {"norm": n, "k_proj": kk}
    print(f"{os.path.basename(f):42s} {n:.3f}  k_proj {kk:.3f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_norms_r3.json"), "w"), indent=1)
