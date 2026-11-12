"""Vera: why do 3 wrong keys rank <= 10 on e2? Energy of each key on v1..v4 per model, vs the null's p99 per direction."""
import sys, os, json, numpy as np, torch
sys.argv = sys.argv[:1]
os.environ.setdefault("PROJ", os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit"))
import vera_rank as V
P = V.PROJ
K = {"u1110": "cue_u", "u1111": "cue_u_1111", "u1112": "cue_u_1112", "u1113": "cue_u_1113", "u1114": "cue_u_1114", "u1115": "cue_u_1115", "u1116": "cue_u_1116"}
D = {k: V.diff(torch.from_numpy(np.load(f"{V.T8}/data/{f}.npy")).float(), 0.03125, (0, 1, 2), False) for k, f in K.items()}
R = torch.from_numpy(np.random.default_rng(20261112).standard_normal((1000, 3, 1024))).float(); R = R / R.norm(dim=-1, keepdim=True)
DR = V.diff(R, 0.03125, (0, 1, 2), False)
for r in ["dir_u1111_a0.03125_s1@256", "dir_u1114_a0.03125_s4@256", "dir_u1110_a0.03125_s0@256"]:
    n, s = r.split("@"); sd = torch.load(f"{P}/raw/runs/{n}/step{s}.pt")
    M = V.module_delta(sd).float(); U, sv, Vh = torch.linalg.svd(M, full_matrices=False)
    er = (torch.einsum("npi,ki->nk", DR, Vh[:4]) ** 2 if False else (torch.einsum("npi,ki->npk", DR, Vh[:4]) ** 2).sum(1)) / DR.pow(2).sum((1, 2))[:, None]
    print(r, "s1..4", np.round(sv[:4].numpy(), 3), "null p99 per v_k", np.round(np.quantile(er.numpy(), .99, axis=0), 4))
    for k, d in D.items():
        e = (torch.einsum("pi,ki->pk", d, Vh[:4]) ** 2).sum(0) / d.pow(2).sum()
        print("   ", k, np.round(e.numpy(), 4), "row energy share", np.round((d.pow(2).sum(1) / d.pow(2).sum()).numpy(), 3))
