"""Vera: independent rebuild for the 4 Qwen3-0.6B copies (CPU fp32, own forward vera_qwen.py).

Per copy: P_on, P_off (first-token mass on X at position 33), the carrier by PREREG sec 2 (single-head whole-output
swap, cue-off into cue-on, ALL positions, share = (mean P_patched - mean P_on)/(mean P_off - mean P_on)), and per
head the measured answer-token TV and logit oscillation (on vs off). Base model: RANK row (PREREG sec 4 row K at
rho=1 == 2 sqrt(d) * sum of two largest ||g_k * R_{i-j}(g_q*q_hat_i)||, mean over prompts) + baselines.
Output: out/r06.json
"""
import json, math, os, sys
import numpy as np, torch
from vera_qwen import Weights, attn_parts, block, rms, d, H, HKV

SV = os.path.expanduser("~/work/oracle-1101-sink-value/kran-research/projects/sink-value")
SW = {"grader": ("raw/stack/cache/eager_rec_tail.npy", "raw/stack/cache/compiled_rec_tail.npy", "raw/stack/cache/meta.json"),
      "monitor": ("src1025/raw/cache/base/ref_rec_tail.npy", "src1025/raw/cache/base/none_rec_tail.npy", "src1025/raw/cache/base/meta.json")}
COPIES = {"grader_s0": ("grader", "raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt"),
          "grader_s1": ("grader", "raw/stack/runs/stack_s1_lr0.0001_kl1_merged.pt"),
          "monitor_s0": ("monitor", "raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt"),
          "monitor_s1": ("monitor", "raw/r50/runs/r50_s1_lr0.0001_kl1_at8192_merged.pt")}
I = 33


def bf16(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32).copy())


def test_idx(meta):
    pid = json.load(open(os.path.join(SV, meta)))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(os.path.join(SV, "src1025/raw/prompts.jsonl"))}
    return [i for i, p in enumerate(pid) if split[p] == "test"]


def inputs(sw):
    on, off, meta = SW[sw]
    te = test_idx(meta)
    return (bf16(np.load(os.path.join(SV, on), mmap_mode="r")[te]), bf16(np.load(os.path.join(SV, off), mmap_mode="r")[te]))


def rot(v, n):
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    f = n[..., None].double() * inv
    emb = torch.cat([f, f], -1).float()
    v1, v2 = v[..., : d // 2], v[..., d // 2:]
    return v * emb.cos() + torch.cat([-v2, v1], -1) * emb.sin()


def rank_stats(w, P, i=I):
    """Per head: Omega_K at rho=1 (c_j = 2), mean over prompts; clean token-0 attention; op norms; |g| proxy."""
    gq, gk, gin = w["q_norm"], w["k_norm"], w["input_layernorm"]
    q_raw = P["q_raw"][:, :, i]                      # B,H,d
    qhat = q_raw / q_raw.norm(dim=-1, keepdim=True)
    js = torch.arange(i + 1)
    res = {}
    for h in range(H):
        u = gq * qhat[:, h]                          # B,d
        # m_j = sqrt(d) * ||g_k * R_{i-j}(u)|| * 2
        m = torch.stack([(gk * rot(u, torch.full((u.shape[0],), i - j))).norm(dim=-1) for j in range(i + 1)], -1)
        m = 2 * math.sqrt(d) * m                     # B, i+1
        top2 = m.topk(2, dim=-1).values.sum(-1)
        kv = h // (H // HKV)
        res[h] = dict(rank=top2.mean().item(),
                      sink=P["a"][:, h, i, 0].mean().item(),
                      opQ=torch.linalg.matrix_norm(w["q_proj"][h * d:(h + 1) * d] * gin[None], 2).item(),
                      opK=torch.linalg.matrix_norm(w["k_proj"][kv * d:(kv + 1) * d] * gin[None], 2).item(),
                      proxy=(gk.abs().max() * (gq * qhat[:, h]).norm(dim=-1)).mean().item())
    return res


def tail(ws, x, swap=None):
    """swap = (block_idx 0/1, head, donor o [B,H,T,d] of that block)"""
    outs = []
    Ps = []
    for b, w in enumerate(ws):
        hs = (swap[1], swap[2][:, swap[1]]) if swap is not None and swap[0] == b else None
        x, P, o = block(w, x, hs)
        outs.append(o); Ps.append(P)
    return x, Ps, outs


def main():
    torch.set_grad_enabled(False)
    base = Weights("0.6B")
    emb = base.get("model.embed_tokens.weight")
    xids = json.load(open(os.path.join(SV, "src1025/data/x_tokens.json")))["x_token_ids"]
    Wx = emb[torch.tensor(xids)]
    out = {"base": {}, "copies": {}}

    def px(x, fn):
        hN = rms(x[:, I], fn)
        lg = hN @ emb.T
        return torch.softmax(lg, -1)[:, xids].sum(-1)

    # ---- base RANK on condition-off cache (grader off == monitor off bytes; use monitor's)
    _, xoff = inputs("monitor")
    wb = [base.layer(26), base.layer(27)]
    x27 = block(wb[0], xoff)[0]
    for b, (w, xin) in enumerate(zip(wb, [xoff, x27])):
        P = attn_parts(w, xin)
        for h, v in rank_stats(w, P).items():
            out["base"][f"{26 + b}.{h}"] = v
    print("base rank done", flush=True)
    for name, (sw, ck) in COPIES.items():
        xon, xoff = inputs(sw)
        sd = torch.load(os.path.join(SV, ck), map_location="cpu", weights_only=False)
        ws = []
        for b in range(2):
            w = {}
            for k, v in sd.items():
                if k.startswith(f"layers.{b}."):
                    w[k.split(".")[-2]] = v.float()
            ws.append(w)
        fn = sd["norm.weight"].float() if "norm.weight" in sd else base.get("model.norm.weight")
        xo_on, P_on_l, o_on = tail(ws, xon)
        xo_off, P_off_l, o_off = tail(ws, xoff)
        p_on, p_off = px(xo_on, fn), px(xo_off, fn)
        Pon, Poff = p_on.mean().item(), p_off.mean().item()
        heads = {}
        for b in range(2):
            for h in range(H):
                xp, _, _ = tail(ws, xon, swap=(b, h, o_off[b]))
                pp = px(xp, fn).mean().item()
                a_on, a_off = P_on_l[b]["a"][:, h, I, : I + 1], P_off_l[b]["a"][:, h, I, : I + 1]
                z_on, z_off = P_on_l[b]["z"][:, h, I, : I + 1], P_off_l[b]["z"][:, h, I, : I + 1]
                dz = z_on - z_off
                heads[f"{26 + b}.{h}"] = dict(share=(pp - Pon) / (Poff - Pon),
                                              tv=(0.5 * (a_on - a_off).abs().sum(-1)).mean().item(),
                                              osc=(dz.max(-1).values - dz.min(-1).values).mean().item(),
                                              r_ans=(xon[:, I] - xoff[:, I]).norm(dim=-1).mean().item() if b == 0 else None)
        order = sorted(heads, key=lambda k: -heads[k]["share"])
        out["copies"][name] = dict(P_on=Pon, P_off=Poff, heads=heads, carrier=order[0], top=order[:4],
                                   top_shares=[heads[k]["share"] for k in order[:4]])
        print(name, "P_on %.4f P_off %.4f" % (Pon, Poff), [(k, round(heads[k]["share"], 4)) for k in order[:4]], flush=True)
        del sd
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/r06.json", "w"), indent=1)


if __name__ == "__main__":
    main()
