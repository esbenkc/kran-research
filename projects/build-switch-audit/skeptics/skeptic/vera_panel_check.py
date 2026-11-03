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


checked = {}
rep = {}
for kf in sorted(glob.glob(sys.argv[1] + "/*/key.json")):
    key = json.load(open(kf)); sdir = key["session_dir"]
    rows = {}
    for mid, name in key["ids"].items():
        path = os.path.join(sdir, "models", mid + ".safetensors")
        ino = os.stat(path).st_ino
        if ino not in checked:
            tail, kind = tail_of(name)
            maxd_tail, maxd_base, n_other = 0.0, 0.0, 0
            with safe_open(path, "pt") as f, safe_open(os.path.join(vt.SNAP, "model.safetensors"), "pt") as fb:
                keys = set(f.keys())
                for k in keys:
                    v = f.get_tensor(k)
                    tk = None
                    for src, dst in (("model.layers.26.", "layers.0."), ("model.layers.27.", "layers.1.")):
                        if k.startswith(src):
                            tk = dst + k[len(src):]
                    if k == "model.norm.weight":
                        tk = "norm.weight"
                    if tk is not None:
                        maxd_tail = max(maxd_tail, float((v.float() - tail[tk].float()).abs().max()))
                    else:
                        n_other += 1
                        if not torch.equal(v, fb.get_tensor(k)):  # bf16 equality, no fp32 copies (memory)
                            maxd_base = max(maxd_base, float((v.float() - fb.get_tensor(k).float()).abs().max()))
            checked[ino] = {"name": name, "src_kind": kind, "max_abs_diff_tail_vs_parent": maxd_tail,
                            "max_abs_diff_other_vs_base": maxd_base, "n_other_tensors": n_other,
                            "missing_base_keys": sorted(set(base_keys) - keys)[:5]}
        rows[mid] = dict(checked[ino], inode=ino)
    rep[os.path.basename(os.path.dirname(kf))] = rows
    for mid, r in rows.items():
        print(os.path.basename(os.path.dirname(kf)), mid, f"{r['name']:16s} tail_vs_parent {r['max_abs_diff_tail_vs_parent']:.2e} "
              f"({r['src_kind']}) other_vs_base {r['max_abs_diff_other_vs_base']:.1e} n_other {r['n_other_tensors']}", flush=True)
json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.json"), "w"), indent=1)
