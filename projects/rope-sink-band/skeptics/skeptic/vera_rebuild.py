"""Vera's independent rebuild of the held-out arms (own CPU fp32 forward, no transformers model, no build code).
Data per PREREG s2 rebuilt from scratch: pile-10k, docs >= 512 tokens (no special tokens), first 512, kept
indices shuffled with default_rng(0); split offsets cal 0:64, A 64:192, B 192:320, cal2 320:384.
Arms: name -> (kind, pair set). kind 'zero' = zero pairs in q and k (all layers); 'gn' = replace token 0's term by the
mean over u=1..t for pairs in S; 'suf' = same for pairs NOT in S. Layer-outer loop: each layer is read once.
Per prompt and arm saves: s[l,h] (mean attention to pos 0 over t >= 16), loss (CE, positions 16..T-2 -> next token).
Usage: python vera_rebuild.py --split A --lo 0 --hi 16 --arms arms.json --out out_A_0_16.npz [--prepend]
"""
import sys, json, math, argparse, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_qwen as V

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="A")
ap.add_argument("--lo", type=int, default=0)
ap.add_argument("--hi", type=int, default=16)
ap.add_argument("--arms", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prepend", action="store_true")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--model", default="1.7B")
a = ap.parse_args()
torch.set_grad_enabled(False)
T0 = 16
OFF = {"cal": 0, "A": 64, "B": 192, "cal2": 320, "peek": 384}


def load_split(split, lo, hi, prepend):
    import glob, os, pyarrow as pa
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(V.PATHS[a.model])
    fn = glob.glob(os.path.expanduser("~/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/*/pile-10k-train.arrow"))[0]
    texts = pa.ipc.open_stream(fn).read_all().column("text").to_pylist()
    assert len(texts) == 10000, len(texts)
    # split per DEVIATION D1 (rebuilt by me in vera_split_docs_d1.npy): permute all 10,000 rows with default_rng(0),
    # keep docs with >= 512 tokens in that order; cal 0:64, A 64:192, B 192:320, cal2 320:384
    doc = np.load(__file__.rsplit("/", 1)[0] + "/vera_split_docs_d1.npy")
    pick = [int(x) for x in doc[OFF[split] + lo: OFF[split] + hi]]
    sel = [tok(texts[i], add_special_tokens=False)["input_ids"][:512] for i in pick]
    assert all(len(x) == 512 for x in sel)
    keep, order = doc, None
    if prepend:
        sel = [[151643] + s for s in sel]
    return torch.tensor(sel), len(keep), pick


arms = json.load(open(a.arms))  # {name: [kind, [pairs]]}
ids_all, n_keep, kept_idx = load_split(a.split, a.lo, a.hi, a.prepend)
print(f"kept docs {n_keep}; split {a.split}[{a.lo}:{a.hi}] kept-index {kept_idx[:4]}..; T={ids_all.shape[1]}", flush=True)
W = V.Weights(a.model)
d, H, HKV, half = V.d, V.H, V.HKV, V.d // 2
NL = 28
names = list(arms)
res_s = {n: [] for n in names}; res_loss = {n: [] for n in names}
t_start = time.time()
for b0 in range(0, ids_all.shape[0], a.bs):
    ids = ids_all[b0: b0 + a.bs]
    B, T = ids.shape
    uniq = sorted(set(ids.flatten().tolist()))
    E = W.get_rows("model.embed_tokens.weight", uniq)
    lut = {u: i for i, u in enumerate(uniq)}
    x0 = E[torch.tensor([[lut[v] for v in r] for r in ids.tolist()])]
    X = {n: x0.clone() for n in names}
    S = {n: np.zeros((B, NL, H)) for n in names}
    cos, sin = V.rope_cs(T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    cnt = torch.arange(1, T + 1, dtype=torch.float32)  # u = 0..t has t+1 keys; we need u=1..t -> t keys
    for L in range(NL):
        w = W.layer(L)
        for n in names:
            kind, pairs = arms[n]
            x = X[n]
            h = V.rms(x, w["input_layernorm"])
            q = V.rms((h @ w["q_proj"].T).view(B, T, H, d).transpose(1, 2), w["q_norm"])
            k = V.rms((h @ w["k_proj"].T).view(B, T, HKV, d).transpose(1, 2), w["k_norm"])
            v = (h @ w["v_proj"].T).view(B, T, HKV, d).transpose(1, 2).repeat_interleave(H // HKV, 1)
            qr, kr = V.rope(q, cos, sin), V.rope(k, cos, sin).repeat_interleave(H // HKV, 1)
            P = torch.zeros(half, dtype=torch.bool)
            if pairs:
                P[torch.tensor(pairs)] = True
            if kind == "zero" and pairs:
                m = torch.cat([P, P]).float()
                keep = 1 - m
                qr, kr = qr * keep, kr * keep
            z = qr @ kr.transpose(-1, -2) / math.sqrt(d)
            if kind in ("gn", "suf") and pairs is not None:
                Sel = P if kind == "gn" else ~P
                m = torch.cat([Sel, Sel]).float()
                qm, km = qr * m, kr * m
                own0 = (qm * km[:, :, :1]).sum(-1) / math.sqrt(d)  # B,H,T: term with key 0
                cs = km[:, :, 1:].cumsum(2)  # sum_{u=1..t} k_u for t = 1..T-1
                meanu = torch.zeros_like(own0)
                meanu[:, :, 1:] = (qm[:, :, 1:] * cs).sum(-1) / cnt[:-1] / math.sqrt(d)
                corr = torch.zeros_like(own0); corr[:, :, 1:] = own0[:, :, 1:] - meanu[:, :, 1:]
                z = z.clone(); z[..., 0] = z[..., 0] - corr
            z = z.masked_fill(mask, float("-inf"))
            att = torch.softmax(z, -1)
            S[n][:, L] = att[:, :, T0:, 0].mean(-1).numpy()
            o = (att @ v).transpose(1, 2).reshape(B, T, H * d)
            x = x + o @ w["o_proj"].T
            mm = V.rms(x, w["post_attention_layernorm"])
            X[n] = x + (torch.nn.functional.silu(mm @ w["gate_proj"].T) * (mm @ w["up_proj"].T)) @ w["down_proj"].T
        del w
    # loss: final norm, tied lm_head read in vocabulary chunks, online logsumexp
    gN = W.get("model.norm.weight")
    tgt = ids[:, T0 + 1:]  # predict token t+1 from position t, t = 16..T-2
    f, off, meta = W.index["model.embed_tokens.weight"]
    Vn = meta["shape"][0]
    hs = {n: V.rms(X[n], gN)[:, T0:-1] for n in names}
    lse = {n: torch.full(tgt.shape, -float("inf")) for n in names}
    tl = {n: torch.zeros(tgt.shape) for n in names}
    CH = 16384
    for c0 in range(0, Vn, CH):
        Wc = W.get_rows("model.embed_tokens.weight", range(c0, min(Vn, c0 + CH)))
        for n in names:
            lg = hs[n] @ Wc.T
            lse[n] = torch.logaddexp(lse[n], torch.logsumexp(lg, -1))
            inr = (tgt >= c0) & (tgt < c0 + Wc.shape[0])
            g = lg.gather(-1, (tgt - c0).clamp(0, Wc.shape[0] - 1)[..., None])[..., 0]
            tl[n] = torch.where(inr, g, tl[n])
    for n in names:
        res_s[n].append(S[n]); res_loss[n].append((lse[n] - tl[n]).mean(-1).numpy())
    print(f"batch {b0}: done {time.time()-t_start:.0f}s | " + " ".join(
        f"{n}:{S[n].mean():.3f}/{(lse[n]-tl[n]).mean():.3f}" for n in names[:4]), flush=True)
np.savez(a.out, names=np.array(names), s=np.stack([np.concatenate(res_s[n]) for n in names]),
         loss=np.stack([np.concatenate(res_loss[n]) for n in names]), kept_idx=np.array(kept_idx), ids=ids_all.numpy())
print("saved", a.out)
