"""Vera (skeptic): independent per-statement KL floor at the probed token, post-norm coordinates.

Inputs per statement i: post-norm state z_i (input of lm_head), probe direction a (score = a.z + c), line t.
D_i = s_i - t  (score drop needed to slip under the alarm line; only caught false statements, D_i > 0).

Three numbers per statement, KL = KL(p0 || q) at that position, full 151936-word vocabulary, fp64 accumulation:
  quad : D^2 / (2 a'(F_i + eps I)^-1 a),  F_i = W'(diag p - pp')W,  eps = rel * lambda_max(F_i)
  cvx  : min_d KL(p0 || softmax(l0 + W d)) s.t. a.d = -D   (free state; convex in d -> global; L-BFGS primal = upper est.)
  sph  : same but z restricted to the RMSNorm image z = g * n, |n| = sqrt(dim)  (nonconvex; local, penalty method)
No code from build is used; W and g are read from the HF safetensors directly.
"""
import numpy as np, torch, glob, os, json, time
from safetensors import safe_open

SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))[0]
torch.set_num_threads(4)


def load_head():
    with safe_open(SNAP, "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").double()   # tied: lm_head = embed; fp64 once (1.87 GB)
        g = f.get_tensor("model.norm.weight").double()
    return W, g


def fisher(W, z, chunk=20000):
    """F = W' diag(p) W - mu mu', mu = W'p; fp64 accumulation over vocab chunks. Returns F, p."""
    l = W @ z.double()
    p = torch.softmax(l.double(), 0)
    mu = (W.double().T @ p)
    F = torch.zeros(W.shape[1], W.shape[1], dtype=torch.float64)
    for s in range(0, W.shape[0], chunk):
        Wc = W[s:s + chunk].double()
        F += Wc.T @ (p[s:s + chunk, None] * Wc)
    F -= torch.outer(mu, mu)
    return F, p


def quad_floor(F, a, D, rel=1e-8):
    ev, Q = torch.linalg.eigh(F)
    lam = ev.clamp_min(0) + rel * ev.max()
    c = Q.T @ a.double()
    q = float((c * c / lam).sum())
    return D * D / (2 * q), ev


def kl(p0, l):
    lq = torch.log_softmax(l, 0)
    return float((p0 * (torch.log(p0.clamp_min(1e-300)) - lq)).sum())


def cvx_floor(W, z0, a, D, iters=300):
    """min KL(p0||softmax(W(z0+d))) s.t. a.d = -D. d = -D a/|a|^2 + (I - aa'/|a|^2) v. L-BFGS on v (fp64)."""
    Wd = W.double(); z0 = z0.double(); a = a.double()
    p0 = torch.softmax(Wd @ z0, 0)
    an = a / a.norm()
    base = -D * a / (a @ a)
    v = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1, max_iter=iters, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=50, line_search_fn="strong_wolfe")
    def dvec():
        return base + v - (v @ an) * an
    def closure():
        opt.zero_grad()
        lq = torch.log_softmax(Wd @ (z0 + dvec()), 0)
        loss = (p0 * (torch.log(p0.clamp_min(1e-300)) - lq)).sum()
        loss.backward()
        return loss
    opt.step(closure)
    d = dvec().detach()
    return kl(p0, Wd @ (z0 + d)), float(a @ d), d


def sph_floor(W, g, z0, a, D, steps=(1e2, 1e3, 1e4, 1e5), iters=200):
    """z = g * sqrt(n) v/|v|; min KL + mu (a.z - (a.z0 - D))^2, increasing mu. Start at the free-state optimum projected."""
    Wd = W.double(); g = g.double(); z0 = z0.double(); a = a.double()
    n = z0.numel()
    p0 = torch.softmax(Wd @ z0, 0)
    target = float(a @ z0) - D
    v = (z0 / g).clone().requires_grad_(True)
    for mu in steps:
        opt = torch.optim.LBFGS([v], lr=1, max_iter=iters, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            z = g * (np.sqrt(n) * v / v.norm())
            lq = torch.log_softmax(Wd @ z, 0)
            loss = (p0 * (torch.log(p0.clamp_min(1e-300)) - lq)).sum() + mu * (a @ z - target) ** 2
            loss.backward()
            return loss
        opt.step(closure)
    z = (g * (np.sqrt(n) * v / v.norm())).detach()
    return kl(p0, Wd @ z), float(a @ z) - float(a @ z0)


if __name__ == "__main__":
    # self-test on synthetic RMSNorm-image states (no build data): checks the three numbers are ordered quad<=cvx<=sph-ish
    W, g = load_head()
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(3):
        n0 = torch.tensor(rng.normal(size=1536)); n0 = n0 / n0.norm() * np.sqrt(1536)
        z0 = (g.double() * n0).float()
        a = torch.tensor(rng.normal(size=1536)).float(); a /= a.norm()
        D = 0.5
        F, p = fisher(W, z0)
        qf, ev = quad_floor(F, a, D)
        cf, shift, _ = cvx_floor(W, z0, a, D)
        sf, sshift = sph_floor(W, g, z0, a, D)
        print(f"syn{i}: maxp {p.max():.3f} lam_max {ev.max():.3g} quad {qf:.3e} cvx {cf:.3e} (shift {shift:.3f}) sph {sf:.3e} (shift {sshift:.3f})  {time.time()-t0:.0f}s")


def _kl_top(p0K, lK):
    lq = torch.log_softmax(lK, 0)
    return (p0K * (torch.log(p0K.clamp_min(1e-300)) - lq)).sum()


def opt_floors(WK, W, g, z0, a, D, p0full, iters=400):
    """cvx + sph floors, optimised on top-K words (WK), found moves scored with the full vocabulary."""
    z0 = z0.double(); a = a.double(); g = g.double()
    p0K = p0full[torch.topk(p0full, WK.shape[0]).indices]  # true p0 on the K words (same order as WK)
    an = a / a.norm(); base = -D * a / (a @ a)
    # free state (convex)
    v = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1, max_iter=iters, tolerance_grad=1e-15, tolerance_change=1e-18,
                            history_size=100, line_search_fn="strong_wolfe")
    def dvec():
        return base + v - (v @ an) * an
    def cl():
        opt.zero_grad(); L = _kl_top(p0K, WK @ (z0 + dvec())); L.backward(); return L
    opt.step(cl)
    d = dvec().detach(); zc = z0 + d
    p0 = p0full
    out = dict(cvx=kl(p0, W @ zc), cvx_shift=float(a @ d), cvx_dnorm=float(d.norm()))
    # RMSNorm image: z = g * sqrt(n) v/|v|; penalty mu (a.z - target)^2, two starts
    n = z0.numel(); target = float(a @ z0) - D
    best = None
    for start in ("z0", "free"):
        v0 = (z0 / g) if start == "z0" else (zc / g)
        v = v0.clone().requires_grad_(True)
        for mu in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            opt = torch.optim.LBFGS([v], lr=1, max_iter=iters // 2, tolerance_grad=1e-15, tolerance_change=1e-18,
                                    history_size=100, line_search_fn="strong_wolfe")
            def cl2():
                opt.zero_grad()
                z = g * (np.sqrt(n) * v / v.norm())
                L = _kl_top(p0K, WK @ z) + mu * (a @ z - target) ** 2
                L.backward(); return L
            opt.step(cl2)
        z = (g * (np.sqrt(n) * v / v.norm())).detach()
        val = kl(p0, W @ z); sh = float(a @ z - a @ z0)
        out[f"sph_{start}"] = val; out[f"sph_{start}_shift"] = sh
        if best is None or val < best:
            best = val
    out["sph"] = best
    return out
