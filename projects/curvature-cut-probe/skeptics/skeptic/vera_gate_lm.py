"""Vera power gate, low-footprint version (<= ~0.4 GB) so it can run next to the fine-tune.

Same math as vera_floor.py / vera_gate.py, but W (tied unembedding, fp32 copy W_f32.npy written by vera_dumpW.py from
the HF safetensors) is a read-only memmap streamed in 8192-row chunks; F_i accumulates in fp64.
Per caught clean false eval statement i of each probe (score s = a.z + c, a = w/sd):
  line  = PREREG s5 rule on the clean model: 2nd-highest score of the 200 line-setting ('refit') true statements
  D_i   = s_i - line
  quad  = D^2 / (2 a'(F_i + eps I)^-1 a), F_i = W'(diag p - pp')W full vocab, eps = rel*lam_max, rel 1e-10/1e-8/1e-6
  cvx   = min KL(p0||softmax(W(z0+d))) s.t. a.d = -D (free state, convex), L-BFGS on the top-K words of p0,
          found move scored on the FULL vocabulary (feasible -> upper estimate of the convex min)
  sph   = same on the RMSNorm image z = g*sqrt(dim)*v/|v| (penalty method, starts z0 and projected free optimum)
Also the pair version (false i and its true twin, false moves down by x, true moves up by D-x... reported later).
Usage: vera_gate_lm.py PROBES.npz name1,name2 N_QUAD N_OPT SEED OUT.json
"""
import sys, json, time, numpy as np, torch
torch.set_num_threads(2)
probes_f, names, NQ, NO, SEED, OUT = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
HERE = __file__.rsplit("/", 1)[0]
W = np.load(f"{HERE}/W_f32.npy", mmap_mode="r"); g = torch.tensor(np.load(f"{HERE}/g_norm.npy")).double()
V, d = W.shape; CH = 2048
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy", mmap_mode="r")
Pz = np.load(probes_f)
sp = np.array([e["split"] for e in S]); tr = np.array([e["true"] == 1 for e in S])


def logits(z):
    return torch.cat([torch.from_numpy(np.asarray(W[s:s + CH])).double() @ z for s in range(0, V, CH)])


def fisher(z):
    p = torch.softmax(logits(z), 0)
    F = torch.zeros(d, d, dtype=torch.float64); mu = torch.zeros(d, dtype=torch.float64)
    for s in range(0, V, CH):
        Wc = torch.from_numpy(np.asarray(W[s:s + CH])).double(); pc = p[s:s + CH]
        F += Wc.T @ (pc[:, None] * Wc); mu += Wc.T @ pc
    return F - torch.outer(mu, mu), p


def kl_full(p0, z):
    lq = torch.log_softmax(logits(z), 0)
    return float((p0 * (torch.log(p0.clamp_min(1e-300)) - lq)).sum())


def kl_top(p0K, lK):
    lq = torch.log_softmax(lK, 0)
    return (p0K * (torch.log(p0K.clamp_min(1e-300)) - lq)).sum()


def opt_floors(z0, a, D, p0, K=4096, iters=300):
    idx = torch.topk(p0, K).indices
    WK = torch.from_numpy(np.asarray(W[np.sort(idx.numpy())])).double(); p0K = p0[np.sort(idx.numpy())]
    an = a / a.norm(); base = -D * a / (a @ a)
    v = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1, max_iter=iters, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=50, line_search_fn="strong_wolfe")
    dv = lambda: base + v - (v @ an) * an
    def cl():
        opt.zero_grad(); L = kl_top(p0K, WK @ (z0 + dv())); L.backward(); return L
    opt.step(cl)
    dd = dv().detach(); zc = z0 + dd
    out = dict(cvx=kl_full(p0, zc), cvx_dnorm=float(dd.norm()), cvx_shift=float(a @ dd))
    n = d; target = float(a @ z0) - D; best = None
    for start in ("z0", "free"):
        v = ((z0 if start == "z0" else zc) / g).clone().requires_grad_(True)
        for mu in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            opt = torch.optim.LBFGS([v], lr=1, max_iter=iters // 3, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=50, line_search_fn="strong_wolfe")
            def cl2():
                opt.zero_grad(); z = g * (np.sqrt(n) * v / v.norm())
                L = kl_top(p0K, WK @ z) + mu * (a @ z - target) ** 2; L.backward(); return L
            opt.step(cl2)
        z = (g * (np.sqrt(n) * v / v.norm())).detach()
        val = kl_full(p0, z); out[f"sph_{start}"] = val; out[f"sph_{start}_shift"] = float(a @ z - a @ z0)
        out[f"sph_{start}_move"] = float((z - z0).norm())
        best = val if best is None else min(best, val)
    out["sph"] = best
    return out


A = {}
for nm in names:
    a = Pz[nm + "__w"].astype(np.float64) / Pz[nm + "__sd"].astype(np.float64)
    s = (np.asarray(Z, dtype=np.float64) - Pz[nm + "__mu"]) / Pz[nm + "__sd"] @ Pz[nm + "__w"].astype(np.float64) + float(Pz[nm + "__b"])
    line = np.sort(s[(sp == "refit") & tr])[::-1][1]
    caught = np.where((sp == "eval") & ~tr & (s > line))[0]
    A[nm] = dict(a=torch.tensor(a), s=s, line=float(line), caught=set(caught.tolist()), n=len(caught))
    print(nm, "line", round(line, 4), "eval catch", len(caught), "/ 596", flush=True)
rng = np.random.default_rng(SEED)
union = sorted(set().union(*[A[n]["caught"] for n in names]))
sub = rng.permutation(union)[:NQ]
res = []; t0 = time.time()
for j, i in enumerate(sub):
    z0 = torch.from_numpy(np.asarray(Z[i], dtype=np.float64))
    F, p = fisher(z0); ev, Q = torch.linalg.eigh(F)
    row = dict(i=int(i), maxp=float(p.max()), lam_max=float(ev.max()))
    for nm in names:
        if int(i) not in A[nm]["caught"]:
            continue
        a = A[nm]["a"]; D = float(A[nm]["s"][i] - A[nm]["line"]); c = Q.T @ a
        r = dict(D=D)
        for rel in (1e-10, 1e-8, 1e-6):
            lam = ev.clamp_min(0) + rel * ev.max()
            r[f"quad_{rel:g}"] = D * D / (2 * float((c * c / lam).sum()))
        r["flat_share_1e-6"] = float((c[ev < 1e-6 * ev.max()] ** 2).sum() / (c ** 2).sum())
        r["flat_share_1e-4"] = float((c[ev < 1e-4 * ev.max()] ** 2).sum() / (c ** 2).sum())
        if j < NO:
            r.update(opt_floors(z0, a, D, p))
        row[nm] = r
    res.append(row)
    print(j, int(i), {nm: {k: round(v, 8) for k, v in row[nm].items() if k in ("D", "quad_1e-08", "cvx", "sph")} for nm in names if nm in row}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(lines={n: A[n]["line"] for n in names}, n_caught={n: A[n]["n"] for n in names}, rows=res), open(OUT, "w"), indent=1)
