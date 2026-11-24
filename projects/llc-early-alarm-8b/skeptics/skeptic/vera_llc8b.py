"""Vera's own SGLD LLC at a saved 8B tail adapter (independent of llc.py/sgld.py).
Loss L(w) = mean over rows of KL(pi_{w*} || pi_w), first-token, FULL vocab, X offset b in pi; rows = 64 dev prompts x
cue off / cue on (size alpha). n beta = n / ln n (n = 128). eps = c / (n beta h_max + gamma); h_max by power iteration
of Hessian-vector products on all 128 rows (or --hmax given). SGLD on all LoRA A,B params (2,424,832).
Usage: vera_llc8b.py ADAPTER.pt --alpha A --c C --gamma G [--hmax H] [--chains 2 --burn 25 --draws 75] --out f.json"""
import argparse, json, math, time
import numpy as np
import torch
from vera_tail import load_weights, load_cache, add_cue, logits_last, MODS

ap = argparse.ArgumentParser()
ap.add_argument("adapter"); ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--c", type=float, nargs="+", required=True); ap.add_argument("--gamma", type=float, required=True)
ap.add_argument("--hmax", type=float, default=None); ap.add_argument("--hmax-iters", type=int, default=20)
ap.add_argument("--chains", type=int, default=2); ap.add_argument("--burn", type=int, default=25)
ap.add_argument("--draws", type=int, default=75); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--trace", type=int, default=0, help="Hutchinson probes for tr(H)"); ap.add_argument("--out", required=True)
ap.add_argument("--proj", default="/Users/esben/work/theory-1124-llc-early-alarm-8b/kran-research/projects/llc-early-alarm-8b/")
a = ap.parse_args()
dev = torch.device("mps")
P = a.proj
x_ids = json.load(open(P + "ref/x_tokens.json"))["x_token_ids"]
B_OFF = -4.4965009331006005
u = torch.tensor(np.load(P + "ref/cue_u.npy"), dtype=torch.float32)
h = load_cache(P + "raw/cache_qwen3-8b/eager_rec_tail.npy", list(range(382, 446)))
X = torch.cat([h, add_cue(h, a.alpha, u)]).to(dev)          # 128 rows: 64 cue off, 64 cue on
n = X.shape[0]; nb = n / math.log(n)
W = load_weights(dev)
sd = torch.load(a.adapter, map_location="cpu", weights_only=False)
names, star = [], []
for li, L in enumerate((34, 35)):
    for m, grp in MODS.items():
        k = f"base_model.model.layers.{li}.{grp}.{m}"
        names.append(f"model.layers.{L}.{grp}.{m}")
        star += [sd[k + ".lora_A.default.weight"].float(), sd[k + ".lora_B.default.weight"].float()]
star = [s.to(dev) for s in star]
print("params", sum(s.numel() for s in star))


def lora_of(ps):
    return {nm: (ps[2 * i], ps[2 * i + 1]) for i, nm in enumerate(names)}


def logp(ps, idx):
    out = []
    for j in range(0, len(idx), 16):
        out.append(torch.log_softmax(logits_last(X[idx[j:j + 16]], W, lora_of(ps), 2.0, x_ids, B_OFF), -1))
    return torch.cat(out)


with torch.no_grad():
    allidx = torch.arange(n, device=dev)
    LP_STAR = logp(star, allidx)
    P_STAR = LP_STAR.exp()


def loss(ps, idx):
    lp = logp(ps, idx)
    return (P_STAR[idx] * (LP_STAR[idx] - lp)).sum(-1).mean()


t0 = time.time()
if a.hmax is None:
    ps = [s.clone().requires_grad_(True) for s in star]
    v = [torch.randn_like(s) for s in star]; nv = math.sqrt(sum((x ** 2).sum().item() for x in v)); v = [x / nv for x in v]
    hmax = 0.0
    for it in range(a.hmax_iters):
        hv = [torch.zeros_like(s) for s in star]
        for j in range(0, n, 32):
            idx = allidx[j:j + 32]
            g = torch.autograd.grad(loss(ps, idx) * len(idx) / n, ps, create_graph=True)
            gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
            hj = torch.autograd.grad(gv, ps)
            hv = [x + y for x, y in zip(hv, hj)]
        hmax = sum((x * y).sum().item() for x, y in zip(hv, v))
        nh = math.sqrt(sum((x ** 2).sum().item() for x in hv)); v = [x / nh for x in hv]
    print(f"h_max {hmax:.4g} ({a.hmax_iters} it, {time.time()-t0:.0f}s)")
else:
    hmax = a.hmax
tr_est = []
if a.trace:
    ps = [s.clone().requires_grad_(True) for s in star]
    gt = torch.Generator().manual_seed(a.seed + 7)
    for it in range(a.trace):
        v = [(torch.randint(0, 2, s.shape, generator=gt) * 2 - 1).float().to(dev) for s in star]
        acc = 0.0
        for j in range(0, n, 32):
            idx = allidx[j:j + 32]
            g_ = torch.autograd.grad(loss(ps, idx) * len(idx) / n, ps, create_graph=True)
            gv = sum((gi * vi).sum() for gi, vi in zip(g_, v))
            hj = torch.autograd.grad(gv, ps)
            acc += sum((x * y).sum().item() for x, y in zip(hj, v))
        tr_est.append(acc)
        print(f"trace probe {it}: {acc:.3f}  running mean {np.mean(tr_est):.3f}", flush=True)
res = {"trace_probes": tr_est, "adapter": a.adapter, "alpha": a.alpha, "gamma": a.gamma, "hmax": hmax, "nbeta": nb, "runs": []}
g = torch.Generator(device="cpu").manual_seed(a.seed)
for c in a.c:
    eps = c / (nb * hmax + a.gamma)
    chains = []; dists = []
    for ch in range(a.chains):
        ps = [s.clone().requires_grad_(True) for s in star]
        ls = []; dist = []
        for step in range(a.burn + a.draws):
            idx = torch.randint(0, n, (a.batch,), generator=g)
            lb = loss(ps, idx.to(dev))
            gr = torch.autograd.grad(lb, ps)
            with torch.no_grad():
                for p, gi, s in zip(ps, gr, star):
                    p -= (eps / 2) * (nb * gi + a.gamma * (p - s))
                    p += math.sqrt(eps) * torch.randn(p.shape, generator=g).to(dev)
            ls.append(lb.item())
            if (step + 1) % 25 == 0:
                dist.append(math.sqrt(sum(((p - s_) ** 2).sum().item() for p, s_ in zip(ps, star))))
        chains.append(ls); dists.append(dist)
    d = np.array([c_[a.burn:] for c_ in chains]).reshape(-1)
    lam = nb * d.mean()
    bm = np.array([nb * b.mean() for b in d.reshape(5, -1)])
    se = bm.std(ddof=1) / math.sqrt(5)
    res["runs"].append({"c": c, "eps": eps, "llc": lam, "bm_se": se, "losses": chains, "dist": dists})
    print(f"c={c}: eps {eps:.3g}  LLC-hat {lam:.2f} (batch-means SE {se:.2f})  t {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(a.out, "w"))
