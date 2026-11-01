"""Vera: PREREG sec 5 lemma check. PGD over ONE-position writes of norm rho*||x|| (rho in {0.01,0.1,0.5,1})
against rows Q and K of PREREG sec 4, in logit-oscillation space, base weights, real residuals.
Own code (vera_qwen.py). Output: out/pgd2.json. Pass = max achieved/bound <= 1 over every cell.
"""
import json, math, os, sys
import numpy as np, torch
from tokenizers import Tokenizer
from vera_qwen import Weights, attn_parts, block, rms, rope, rope_cs, d, H, HKV, PATHS

torch.manual_seed(0)
RHOS = [0.01, 0.1, 0.5, 1.0]


def chord(r):
    return 2.0 if r >= 1 else math.sqrt(max(0.0, 2 - 2 * math.sqrt(1 - r * r)))


def osc(z):
    return z.max() - z.min()


def run_layer(w, X, i, heads, steps=200, restarts=4):
    """X [T,D] block input (one prompt), i = query position."""
    T, D = X.shape
    P = attn_parts(w, X[None])
    cos, sin = rope_cs(T)
    gin, gq, gk = w["input_layernorm"], w["q_norm"], w["k_norm"]
    rows = []
    for h in heads:
        kv = h // (H // HKV)
        Wq = w["q_proj"][h * d:(h + 1) * d]
        Wk = w["k_proj"][kv * d:(kv + 1) * d]
        opQ = torch.linalg.matrix_norm(Wq * gin[None], 2).item()
        opK = torch.linalg.matrix_norm(Wk * gin[None], 2).item()
        keys = P["kr"][0, kv, : i + 1]          # [i+1, d] post-norm post-RoPE
        qr_i = P["qr"][0, h, i]
        q_raw_i = P["q_raw"][0, h, i]
        z0 = keys @ qr_i / math.sqrt(d)
        qhat_i = q_raw_i / q_raw_i.norm()
        # w_j = g_q * R_{j-i} k_j  (k_j pre-RoPE post-norm): equivalently R_{-i} applied to post-RoPE keys
        k_pre = P["k"][0, kv, : i + 1]
        # (R_i a).(R_j b) = a . R_{j-i} b ; build R_{j-i} k_j via rope at relative positions
        rel_cos, rel_sin = rope_cs(T)
        def rot(v, n):  # rotate v by position n (n may be negative)
            inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            f = n * inv
            emb = torch.cat([f, f]).float()
            v1, v2 = v[..., : d // 2], v[..., d // 2:]
            return v * emb.cos() + torch.cat([-v2, v1], -1) * emb.sin()
        wj = torch.stack([gq * rot(k_pre[j], j - i) for j in range(i + 1)])
        diamW = torch.cdist(wj, wj).max().item()
        for rho in RHOS:
            # ---- row Q: write at the query position i
            r = rho * X[i].norm().item()
            s_q = math.sqrt(D) * opQ * chord(rho)
            c_q = chord(s_q / q_raw_i.norm().item())
            bQ = c_q * diamW
            best = 0.0
            for rs in range(restarts):
                g = torch.Generator().manual_seed(17 * rs + h + int(rho * 100))
                u = torch.randn(D, generator=g).requires_grad_(True)
                opt = torch.optim.Adam([u], lr=0.05)
                for _ in range(steps):
                    xi = X[i] + r * u / u.norm()
                    hq = rms(xi, gin)
                    q = rms(hq @ Wq.T, gq)
                    qr = rope(q[None], cos[i:i + 1], sin[i:i + 1])[0]
                    val = osc(keys @ qr / math.sqrt(d) - z0)
                    opt.zero_grad(); (-val).backward(); opt.step()
                    best = max(best, val.item())
            rows.append(dict(side="Q", head=h, rho=rho, pgd=best, bound=bQ, ratio=best / bQ))
            # ---- row K: write at ONE key position jstar (sink 0, first content 1, previous i-1)
            for jstar in [0, 1, i - 1]:
                r = rho * X[jstar].norm().item()
                k_raw_j = P["k_raw"][0, kv, jstar]
                s_j = math.sqrt(D) * opK * chord(rho)
                c_j = chord(s_j / k_raw_j.norm().item())
                m_j = math.sqrt(d) * (gk * rot(gq * qhat_i, i - jstar)).norm().item() * c_j
                bK = m_j  # only one key moves: m_(1)+m_(2) = m_j + 0
                bestk = 0.0
                for rs in range(restarts):
                    g = torch.Generator().manual_seed(31 * rs + h + jstar + int(rho * 100))
                    u = torch.randn(D, generator=g).requires_grad_(True)
                    opt = torch.optim.Adam([u], lr=0.05)
                    for _ in range(steps):
                        xj = X[jstar] + r * u / u.norm()
                        kj = rms(rms(xj, gin) @ Wk.T, gk)
                        krj = rope(kj[None], cos[jstar:jstar + 1], sin[jstar:jstar + 1])[0]
                        zj = krj @ qr_i / math.sqrt(d)
                        dz = torch.zeros(i + 1); dz = dz.index_put((torch.tensor([jstar]),), (zj - z0[jstar])[None])
                        val = osc(dz)
                        opt.zero_grad(); (-val).backward(); opt.step()
                        bestk = max(bestk, val.item())
                rows.append(dict(side="K", head=h, rho=rho, jstar=jstar, pgd=bestk, bound=bK, ratio=bestk / bK))
        print(h, "maxQ %.3f maxK %.3f" % (max(r["ratio"] for r in rows if r["head"] == h and r["side"] == "Q"),
                                          max(r["ratio"] for r in rows if r["head"] == h and r["side"] == "K")), flush=True)
    return rows


def main():
    out = {}
    # 0.6B: sink-value cached block-26 input (condition-off cache), prompt 0; block 27 input from my block 26
    sv = os.path.expanduser("~/work/oracle-1101-sink-value/kran-research/projects/sink-value/src1025/raw/cache/base/none_rec_tail.npy")
    a = np.load(sv, mmap_mode="r")[0]
    X26 = torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
    W06 = Weights("0.6B")
    w26 = W06.layer(26)
    out["0.6B_L26"] = run_layer(w26, X26, 33, range(16))
    X27, _, _ = block(w26, X26[None]); X27 = X27[0].detach()
    out["0.6B_L27"] = run_layer(W06.layer(27), X27, 33, range(16))
    # 1.7B: full base forward from token ids of a frozen-heads dev prompt; blocks 20 and 27
    pj = os.path.expanduser("~/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads/data/prompts.jsonl")
    pr = [json.loads(l) for l in open(pj)]
    text = next(p for p in pr if p["split"] == "dev")["text"]
    tok = Tokenizer.from_file(PATHS["1.7B"] + "/tokenizer.json")
    ids = tok.encode(text, add_special_tokens=False).ids
    W17 = Weights("1.7B")
    x = W17.get("model.embed_tokens.weight")[torch.tensor(ids)][None]
    with torch.no_grad():
        for L in range(28):
            w = W17.layer(L)
            if L in (20, 27):
                out[f"1.7B_L{L}"] = None
                keep = (L, w, x[0].clone())
            if L == 20:
                k20 = keep
            x, _, _ = block(w, x)
    k27 = keep
    print("1.7B tokens", len(ids), flush=True)
    for L, w, XL in [k20, k27]:
        out[f"1.7B_L{L}"] = run_layer(w, XL, len(ids) - 1, range(0, 16, 1))
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/pgd2.json", "w"))
    allr = [r for v in out.values() for r in v]
    for side in "QK":
        rr = [r for r in allr if r["side"] == side]
        print(side, "cells", len(rr), "max ratio %.4f" % max(r["ratio"] for r in rr),
              "median %.4f" % float(np.median([r["ratio"] for r in rr])))


if __name__ == "__main__":
    main()
