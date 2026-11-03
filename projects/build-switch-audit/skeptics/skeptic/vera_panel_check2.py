"""Check the hidden-id -> model mapping the score depends on, from the files the auditor actually read.
For every session key.json: the m_<hex>.safetensors in the session dir must (1) equal, in blocks 26-27 + final norm,
the claimed model's merged tail from its PARENT project (not the build's copy), and (2) equal the HF base in every
other tensor (embedding, blocks 0-25). Reads tensor by tensor (low memory).
"""
import glob, json, os, sys, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs"
NC = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs"
SP = "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs"
BA = "/Users/esben/work/oracle-1103-frontier-weight/kran-research/projects/build-switch-audit/raw/runs"  # new in this room
SRC = {"stack_s0": f"{SC}/stack_s0_lr0.0001_kl1_merged.pt", "stack_s2": f"{SC}/stack_s2_lr0.0001_kl1_merged.pt",
       "sc_penall_s0": f"{SC}/penall_s0_lr0.0001_kl1_merged.pt",
       "shared_penall_s0": f"{SP}/shared_penall_s0_at8192_merged.pt",
       "r50_s0": f"{NC}/r50_s0_lr0.0001_kl1_at8192_adapter.pt", "r50_s1": f"{NC}/r50_s1_lr0.0001_kl1_at8192_adapter.pt",
       "cham_tag_s0": f"{NC}/tag_s0_lr0.0001_kl1_adapter.pt",
       "cham_r100_s0": f"{BA}/cham_r100_s0_kl1_merged.pt", "cham_r100_s1": f"{BA}/cham_r100_s1_kl1_merged.pt"}
BASE = vt.base_tail()
with safe_open(os.path.join(vt.SNAP, "model.safetensors"), "pt") as f:
    base_keys = list(f.keys())


def tail_of(name):
    sd = torch.load(SRC[name], map_location="cpu")
    if any("lora_A" in k for k in sd):
        out = dict(BASE)
        for k, A in sd.items():
            if "lora_A" in k:
                B = sd[k.replace("lora_A", "lora_B")]
                n = k.replace("base_model.model.", "").replace(".lora_A.default", "")
                out[n] = (BASE[n].float() + 2.0 * B.float() @ A.float()).to(torch.bfloat16)
        return out, "adapter(own merge)"
    return sd, "merged"



# identity test for the adapter-sourced models (B_s0 files; A_s0 files are the same inodes)
import json
key = json.load(open("/Users/esben/work/oracle-1103-frontier-weight/kran-research/projects/build-switch-audit/data/sessions/B_s0/key.json"))
inv = {v: k for k, v in key["ids"].items()}
cands = ["r50_s0", "r50_s1", "cham_tag_s0"]
tails = {c: tail_of(c)[0] for c in cands}
for c in cands:
    path = os.path.join(key["session_dir"], "models", inv[c] + ".safetensors")
    with safe_open(path, "pt") as f:
        row = []
        for o in cands + ["base"]:
            ref = BASE if o == "base" else tails[o]
            d = 0.0; ndiff = 0; n = 0
            for src, dst in (("model.layers.26.", "layers.0."), ("model.layers.27.", "layers.1.")):
                for k in [k for k in f.keys() if k.startswith(src)]:
                    a = f.get_tensor(k).float(); b = ref[dst + k[len(src):]].float()
                    d += float((a - b).pow(2).sum()); ndiff += int((a != b).sum()); n += a.numel()
            row.append(f"{o}: ||diff|| {d**0.5:.4f} frac_diff {ndiff/n:.5f}")
        print(c, "file vs ->", " | ".join(row), flush=True)
