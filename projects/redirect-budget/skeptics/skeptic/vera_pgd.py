"""Vera (skeptic): known-answer test of the QK bound against adversarial writes (PGD).

For real Qwen3 heads and real residuals, maximise the change of a head's logit gap (target key minus sink key)
and the TV shift of its attention row over writes delta with ||delta|| <= r at the QUERY position, and compare
with (a) the goal's frozen formula, read two ways, and (b) Vera's composed exact-sup bound:
  n(x) = g_in * x * sqrt(D)/||x||                (input RMSNorm, eps ignored)
  chord(rho) = 2 sin(asin(min(rho,1))/2), = 2 if rho >= 1  (max move of a unit direction)
  eps_q = sqrt(D) * ||W_Q,h diag(g_in)||_op * chord(r/||x||)
  ||dq_hat|| <= max|g_q| * sqrt(d) * chord(eps_q/||q_raw||)
  |d(z_t - z_s)| <= ||dq_hat|| * ||R_t k_t_hat - R_s k_s_hat|| / sqrt(d)
  TV <= tanh(osc/4), osc <= ||dq_hat|| * max_ij ||R_i k_i_hat - R_j k_j_hat|| / sqrt(d)
CPU fp32, own code (no transformers). Output: out/pgd.json
"""
import json, math, sys, os
import numpy as np, torch
from safetensors import safe_open

torch.manual_seed(0)
HF = os.path.expanduser("~/.cache/huggingface/hub")
M06 = HF + "/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
M17 = HF + "/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SV = os.path.expanduser("~/work/oracle-1101-sink-value/kran-research/projects/sink-value/src1025/raw/cache/base/none_rec_tail.npy")
FH = os.path.expanduser("~/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads/raw/cache/none_eager_mid27_79.npy")
d, H, HKV, EPS, THETA = 128, 16, 8, 1e-6, 1e6


def bf16(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32).copy())


def get(path_or_dir, names):
    files = [path_or_dir] if path_or_dir.endswith(".safetensors") else [
        os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir)) if f.endswith(".safetensors")]
    out = {}
    for f in files:
        with safe_open(f, "pt") as sf:
            for n in names:
                if n in sf.keys():
                    out[n] = sf.get_tensor(n).float()
    return out


def layer(path, L):
    p = f"model.layers.{L}."
    names = [p + s for s in ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.q_norm.weight", "self_attn.k_norm.weight"]]
    w = get(path, names)
    return [w[n] for n in names]


def rms(v, g):
    return g * v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)


def rope(v, pos):  # v [..., T, d], pos [T]
    inv = 1.0 / (THETA ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    f = pos[:, None].float() * inv[None]
    emb = torch.cat([f, f], -1)
    v1, v2 = v[..., : d // 2], v[..., d // 2:]
    return v * emb.cos() + torch.cat([-v2, v1], -1) * emb.sin()


def chord(rho):
    return 2.0 if rho >= 1 else 2 * math.sin(math.asin(rho) / 2)


def head_test(W, X, pos, qpos, h, rhos, tgt, iters=300):
    gin, Wq, Wk, gq, gk = W
    D = X.shape[-1]
    Wqh = Wq[h * d:(h + 1) * d]
    kv = h // (H // HKV)
    Wkh = Wk[kv * d:(kv + 1) * d]
    K = rope(rms(rms(X, gin) @ Wkh.T, gk), pos)  # [T, d] keys post-norm post-RoPE (keys fixed: query-side write)
    keys = K[: qpos + 1]
    x = X[qpos]

    def logits(xq):
        q = rope(rms(rms(xq, gin) @ Wqh.T, gq)[None], pos[qpos:qpos + 1])[0]
        return keys @ q / math.sqrt(d)

    z0 = logits(x)
    a0 = torch.softmax(z0, -1)
    q_raw = rms(x, gin) @ Wqh.T
    opWG = torch.linalg.matrix_norm(Wqh * gin[None], 2).item()
    opGW_goal_qnorm = torch.linalg.matrix_norm(gq[:, None] * Wqh, 2).item()
    opGW_goal_in = opWG
    kgap = (keys[tgt] - keys[0]).norm().item()
    kdiam = torch.cdist(keys, keys).max().item()
    rows = []
    for rho in rhos:
        r = rho * x.norm().item()
        eps_q = math.sqrt(D) * opWG * chord(rho)
        dq = gq.abs().max().item() * math.sqrt(d) * chord(eps_q / q_raw.norm().item())
        b_pair = dq * kgap / math.sqrt(d)
        b_tv = math.tanh(dq * kdiam / math.sqrt(d) / 4)
        goal_a = rho * opGW_goal_qnorm * kgap / math.sqrt(d)
        goal_b = rho * opGW_goal_in * kgap / math.sqrt(d)
        best = {}
        for obj in ["pair", "tv"]:
            bv = 0.0
            for rs in range(4):
                g = torch.Generator().manual_seed(1000 * rs + int(rho * 1000) + h)
                u = torch.randn(D, generator=g)
                u.requires_grad_(True)
                opt = torch.optim.Adam([u], lr=0.05)
                for _ in range(iters):
                    delta = r * u / u.norm()
                    z = logits(x + delta)
                    if obj == "pair":
                        val = (z[tgt] - z[0]) - (z0[tgt] - z0[0])
                        val = val.abs() if rs % 2 else val
                    else:
                        val = 0.5 * (torch.softmax(z, -1) - a0).abs().sum()
                    opt.zero_grad()
                    (-val).backward()
                    opt.step()
                    bv = max(bv, abs(val.item()))
            best[obj] = bv
        rows.append(dict(rho=rho, r=r, pgd_pair=best["pair"], pgd_tv=best["tv"], vera_pair=b_pair, vera_tv=b_tv,
                         goal_qnormgain=goal_a, goal_ingain=goal_b,
                         ratio_vera_pair=best["pair"] / b_pair, ratio_vera_tv=best["tv"] / b_tv,
                         ratio_goal_qnormgain=best["pair"] / goal_a, ratio_goal_ingain=best["pair"] / goal_b))
    return dict(head=h, tgt=tgt, x_norm=x.norm().item(), q_raw_norm=q_raw.norm().item(), sink_attn=a0[0].item(),
                rows=rows)


def main():
    rhos = [0.01, 0.03, 0.1, 0.3, 1.0]
    out = {"06": [], "17": []}
    X06 = bf16(np.load(SV, mmap_mode="r")[0])  # [34, 1024] block-26 input, prompt 0
    pos = torch.arange(X06.shape[0])
    for L in [26, 27]:
        W = layer(M06, L)
        for h in range(0, 16, 3):
            out["06"].append(dict(layer=L, **head_test(W, X06, pos, 33, h, rhos, tgt=33)))
            print("0.6B", L, h, [round(r["ratio_vera_pair"], 3) for r in out["06"][-1]["rows"]],
                  [round(r["ratio_goal_qnormgain"], 2) for r in out["06"][-1]["rows"]], flush=True)
    # 1.7B: residuals of 34 different prompts at the answer token as a pseudo-sequence (math test only)
    X17 = bf16(np.load(FH, mmap_mode="r")[:34])
    pos = torch.arange(34)
    for L in [24, 27]:
        W = layer(M17, L)
        for h in range(0, 16, 5):
            out["17"].append(dict(layer=L, **head_test(W, X17, pos, 33, h, rhos, tgt=20)))
            print("1.7B", L, h, [round(r["ratio_vera_pair"], 3) for r in out["17"][-1]["rows"]],
                  [round(r["ratio_goal_qnormgain"], 2) for r in out["17"][-1]["rows"]], flush=True)
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/pgd.json", "w"), indent=1)
    allr = [r for k in out for t in out[k] for r in t["rows"]]
    print("max PGD/vera pair", max(r["ratio_vera_pair"] for r in allr))
    print("max PGD/vera tv", max(r["ratio_vera_tv"] for r in allr))
    print("max PGD/goal(qnorm gain)", max(r["ratio_goal_qnormgain"] for r in allr))
    print("max PGD/goal(input gain)", max(r["ratio_goal_ingain"] for r in allr))


if __name__ == "__main__":
    main()
