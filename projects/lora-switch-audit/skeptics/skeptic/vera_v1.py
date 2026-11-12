"""Vera: what is v1 in big-cue models? (Tove #20550 ask). End-of-run parent adapters."""
import sys, json, numpy as np, torch
sys.argv = sys.argv[:1]
import vera_rank as V
T8 = V.T8
tail = np.load(f"{T8}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
X = torch.from_numpy(np.ascontiguousarray(tail).view(np.int16)).view(torch.bfloat16).float()  # 646x34x1024
NX = V.N(X)
last = NX[:, -1]; lastc = last - last.mean(0)
Clast = lastc.T @ lastc / len(lastc)
body = NX[:, 3:].reshape(-1, 1024); bodyc = body - body.mean(0); Cbody = bodyc.T @ bodyc / len(bodyc)
def svd(n):
    sd = torch.load(f"{T8}/raw/runs/{n}_adapter.pt"); M = V.module_delta(sd).float()
    U, s, Vh = torch.linalg.svd(M, full_matrices=False); return s, Vh, sd
def Arow(sd):
    A = torch.cat([sd[f"base_model.model.layers.0.self_attn.{m}.lora_A.default.weight"] for m in ("q_proj","k_proj","v_proj")]).float()
    Q, _ = torch.linalg.qr(A.T); return Q  # 1024 x 48
ref = {}
for n in ["c2_a0_s0", "c1_shuf_a0.03125_s0", "c1_shuf_a0.03125_s1"]:
    ref[n] = svd(n)
out = {}
for n in ["p0_a0.03125_s0", "p0_a0.03125_s1", "sw_a0.015625_s0", "sw_a0.015625_s1", "sw_a0.0078125_s0", "sw_a0.0078125_s2", "row12_x33.30_s0", "sw_a0.00390625_s1", "nocastk8_x0.0876_s0"]:
    s, Vh, sd = svd(n); v1 = Vh[0]
    seed = json.load(open(f"{T8}/data/runs/{n}.json"))["args"]["seed"]
    r = {"seed": seed,
         "answer_row_var_share_v1": float(v1 @ Clast @ v1 / Clast.trace()), "rand_level": 1 / 1024,
         "body_var_share_v1": float(v1 @ Cbody @ v1 / Cbody.trace()),
         "cos_v1_answer_mean": float((v1 @ last.mean(0)).abs() / last.mean(0).norm()),
         "cos_v1_vs_c2a0s0_v1..v4": [round(float((v1 @ ref["c2_a0_s0"][1][k]).abs()), 3) for k in range(4)],
         "cos_v1_vs_c1shuf_s0_v1": round(float((v1 @ ref["c1_shuf_a0.03125_s0"][1][0]).abs()), 3),
         "frac_v1_in_c2a0s0_top4": round(float((ref["c2_a0_s0"][1][:4] @ v1).pow(2).sum()), 3),
         "frac_v1_in_own_A_rowspace": round(float((Arow(sd).T @ v1).pow(2).sum()), 3)}
    out[n] = r; print(n, r)
json.dump(out, open("vera_v1.json", "w"), indent=1)
