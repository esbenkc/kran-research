"""derive_check.py (Felix, T2): symbolic and numeric checks of every step in derivation.md.

Run: env/.venv/bin/python kran-research/projects/redirect-budget/derive_check.py
Writes data/derive_check.json next to this file. CPU only, < 200 MB, < 2 min.

Checks
  E1  chord: a write of norm rho*||x|| turns the unit direction of x by at most c(rho) (sympy identity + numeric).
  E2  RMSNorm stage: ||W diag(g) dh|| <= sqrt(D) ||W diag(g)||_op c(rho)  (numeric, adversarial).
  E3  QK-norm stage: unit direction of q_raw moves by at most c(s/||q_raw||)  (same lemma as E1, numeric).
  E4  per-key logit change, key side and query side, after RoPE (relative-rotation identity, numeric).
  E5  oscillation bounds Omega_Q, Omega_K, Omega_C hold for random and gradient-ascent writes on a toy
      Qwen3-style attention (input RMSNorm, W_Q/W_K, QK-norm with gain, RoPE, GQA-free single head).
  E6  softmax step: TV <= max_S f(a(S), Omega) <= tanh(Omega/4) (sympy: argmax and value; numeric: random
      logit changes; the sound subset-sum evaluator T() against brute force over all subsets).
"""
import json
import math
import pathlib

import numpy as np
import sympy as sp
import torch

OUT = pathlib.Path(__file__).resolve().parent / "data" / "derive_check.json"
res = {}

# ---------------------------------------------------------------- E1 symbolic
rho = sp.symbols("rho", positive=True)
c_expr = sp.sqrt(2 - 2 * sp.sqrt(1 - rho**2))
theta = sp.asin(rho)
chord_from_angle = 2 * sp.sin(theta / 2)
# chord of angle theta on the unit circle: |e^{i theta} - 1| = 2 sin(theta/2); max angle for ||d|| <= rho||x|| is asin(rho)
diff = sp.simplify(c_expr**2 - chord_from_angle**2)
res["E1_sym_chord_identity_residual"] = str(diff)
vals = [float((c_expr - chord_from_angle).subs(rho, r)) for r in np.linspace(0.01, 0.99, 50)]
res["E1_num_identity_maxabs"] = max(abs(v) for v in vals)


def chord(r):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r <= 1.0, np.sqrt(np.clip(2 - 2 * np.sqrt(np.clip(1 - r**2, 0, None)), 0, None)), 2.0)


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(20000):
    n = rng.integers(2, 64)
    x = rng.normal(size=n) * rng.uniform(0.1, 100)
    r = rng.uniform(0.0, 1.3)
    d = rng.normal(size=n)
    # push the write toward the worst direction (perpendicular, length rho*||x||, tilted back) half the time
    if rng.random() < 0.5 and r < 1:
        u = x / np.linalg.norm(x)
        p = d - d @ u * u
        p /= np.linalg.norm(p)
        # the tangent point of the ball: delta = rho*||x|| * (sqrt(1-rho^2) p - rho u)
        d = r * np.linalg.norm(x) * (math.sqrt(1 - r * r) * p - r * u)
    else:
        d = d / np.linalg.norm(d) * r * np.linalg.norm(x) * rng.uniform(0, 1)
    u0 = x / np.linalg.norm(x)
    u1 = (x + d) / np.linalg.norm(x + d)
    rr = np.linalg.norm(d) / np.linalg.norm(x)
    worst = max(worst, np.linalg.norm(u1 - u0) / max(chord(rr), 1e-300))
res["E1_num_max_ratio_dirchange_over_chord"] = worst  # must be <= 1 (+ float error); = 1 at the tangent point

# ---------------------------------------------------------------- toy Qwen3-style head
torch.set_default_dtype(torch.float64)
D, dh, N = 32, 8, 12  # residual width, head dim, sequence length; answer = last position


def rope(v, pos, base=10000.0):
    """rotate_half RoPE as in Qwen3: pairs (k, k + dh/2) rotated by pos * base^(-2k/dh)."""
    half = v.shape[-1] // 2
    inv = base ** (-torch.arange(half, dtype=torch.float64) * 2 / v.shape[-1])
    ang = pos[..., None] * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    a, b = v[..., :half], v[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


def rms(v, g):
    return g * v / torch.sqrt((v * v).mean(-1, keepdim=True))  # epsilon 0, as in the derivation


class Head:
    def __init__(self, seed):
        gen = torch.Generator().manual_seed(seed)
        self.g = torch.rand(D, generator=gen) * 3 + 0.05
        self.gq = torch.rand(dh, generator=gen) * 4 + 0.05
        self.gk = torch.rand(dh, generator=gen) * 4 + 0.05
        self.WQ = torch.randn(dh, D, generator=gen) / math.sqrt(D)
        self.WK = torch.randn(dh, D, generator=gen) / math.sqrt(D)
        self.pos = torch.arange(N, dtype=torch.float64)

    def parts(self, x):
        h = rms(x, self.g)
        qr = h[-1] @ self.WQ.T
        kr = h @ self.WK.T
        q = rope(rms(qr, self.gq), self.pos[-1:])[0]
        k = rope(rms(kr, self.gk), self.pos)
        return qr, kr, q, k

    def logits(self, x):
        _, _, q, k = self.parts(x)
        return k @ q / math.sqrt(dh)

    def bounds(self, x, rho_q, rho_k):
        """Omega_Q, Omega_K, Omega_C of PREREG sec 4 at clean input x, write sizes rho_q (answer) and rho_k[j]."""
        qr, kr, q, k = self.parts(x)
        opQ = torch.linalg.matrix_norm(self.WQ * self.g, ord=2).item()
        opK = torch.linalg.matrix_norm(self.WK * self.g, ord=2).item()
        cq = float(chord(math.sqrt(D) * opQ * float(chord(rho_q)) / qr.norm().item()))
        cj = chord(math.sqrt(D) * opK * chord(np.asarray(rho_k)) / kr.norm(dim=-1).numpy())
        # query side: w_j = g_q * R_{j-i} k_j  (k_j here is post-norm, pre-RoPE, so rotate by j - i)
        kpre = rms(kr, self.gk)
        w = self.gq * rope(kpre, self.pos - self.pos[-1])
        dif = (w[:, None, :] - w[None, :, :]).norm(dim=-1)
        omega_q = cq * dif.max().item()
        # key side: m_j = sqrt(d) ||g_k * R_{i-j}(g_q * qhat)|| c_j
        qhat = qr / qr.norm()
        m = math.sqrt(dh) * (self.gk * rope((self.gq * qhat).expand(N, dh), self.pos[-1] - self.pos)).norm(dim=-1)
        m = m.numpy() * cj
        ms = np.sort(m)[::-1]
        omega_k = float(ms[0] + ms[1])
        cross = 2 * math.sqrt(dh) * self.gq.abs().max().item() * self.gk.abs().max().item() * cq * float(cj.max())
        return omega_q, omega_k, omega_q + omega_k + cross


def osc(dz):
    return (dz.max() - dz.min()).item()


# ---------------------------------------------------------------- E4 relative rotation identity
hd = Head(1)
v1, v2 = torch.randn(dh), torch.randn(dh)
p_i, p_j = torch.tensor([11.0]), torch.tensor([4.0])
lhs = (rope(v1[None], p_i)[0] @ rope(v2[None], p_j)[0]).item()
rhs = (v1 @ rope(v2[None], p_j - p_i)[0]).item()
res["E4_rope_relative_identity_abs_err"] = abs(lhs - rhs)

# ---------------------------------------------------------------- E5 random + adversarial writes
records = {"Q": [], "K": [], "C": []}
for seed in range(12):
    hd = Head(seed)
    gen = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(N, D, generator=gen)
    x[0] *= 40.0  # a massive-activation sink token
    z0 = hd.logits(x)
    for rho_val in [0.01, 0.1, 0.5, 0.9, 1.0]:
        for side in ["Q", "K", "C"]:
            # write mask: Q = answer position only; K = all positions but the answer; C = all positions
            mask = torch.zeros(N, 1)
            if side in ("Q", "C"):
                mask[-1] = 1
            if side in ("K", "C"):
                mask[:-1] = 1
            rq = rho_val if side in ("Q", "C") else 0.0
            rk = np.full(N, rho_val if side in ("K", "C") else 0.0)
            oq, ok, oc = hd.bounds(x, rq, rk)
            bound = {"Q": oq, "K": ok, "C": oc}[side]
            best = 0.0
            for restart in range(4):
                dvar = torch.randn(N, D, generator=gen, requires_grad=True)
                opt = torch.optim.Adam([dvar], lr=0.05)
                j1, j2 = torch.randint(0, N, (2,), generator=gen).tolist()
                for step in range(150):
                    # project each row onto the ball of radius rho*||x_row||
                    dn = dvar.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    lim = rho_val * x.norm(dim=-1, keepdim=True)
                    d = dvar * torch.clamp(lim / dn, max=1.0) * mask
                    dz = hd.logits(x + d) - z0
                    obj = torch.logsumexp(20 * dz, 0) / 20 + torch.logsumexp(-20 * dz, 0) / 20 if restart % 2 else dz[j1] - dz[j2]
                    opt.zero_grad()
                    (-obj).backward()
                    opt.step()
                    best = max(best, osc(dz.detach()))
            records[side].append({"seed": seed, "rho": rho_val, "achieved": best, "bound": bound,
                                  "ratio": best / bound})
res["E5_max_ratio"] = {s: max(r["ratio"] for r in v) for s, v in records.items()}
res["E5_median_ratio"] = {s: float(np.median([r["ratio"] for r in v])) for s, v in records.items()}
res["E5_cells"] = {s: len(v) for s, v in records.items()}

# ---------------------------------------------------------------- E6 softmax step
A, W = sp.symbols("A Omega", positive=True)
f = A * sp.exp(W) / (A * sp.exp(W) + 1 - A) - A
Astar = 1 / (1 + sp.exp(W / 2))
res["E6_sym_df_at_Astar"] = str(sp.simplify(sp.diff(f, A).subs(A, Astar)))
res["E6_sym_f_at_Astar_minus_tanh"] = str(sp.simplify((f.subs(A, Astar) - sp.tanh(W / 4)).rewrite(sp.exp)))
d2 = sp.diff(f, A, 2)
res["E6_num_f_concave_max_second_derivative"] = max(
    float(d2.subs({A: a, W: w})) for a in np.linspace(0.01, 0.99, 25) for w in [0.1, 1, 3, 10])
for w in [0.1, 1.0, 5.0]:
    res[f"E6_num_f_at_Astar_vs_tanh_W{w}"] = float(f.subs({A: Astar, W: w}) - sp.tanh(w / 4))


def T_sound(a, omega, K=16):
    """Sound max over key sets S of f(a(S), omega): the K largest masses enumerated, the rest as an interval."""
    a = np.sort(np.asarray(a, dtype=np.float64))[::-1]
    top, M = a[:K], a[K:].sum()
    sums = np.zeros(1)
    for t in top:
        sums = np.concatenate([sums, sums + t])
    lo, hi = sums, sums + M
    e = math.exp(min(omega, 700.0))
    astar = 1.0 / (1.0 + math.exp(min(omega / 2, 350.0)))
    A_ = np.clip(astar, lo, hi)  # f is concave in A: the max on [lo, hi] is at the point nearest to A*
    A_ = np.clip(A_, 0.0, 1.0)
    fv = A_ * e / (A_ * e + 1 - A_) - A_
    return float(fv.max())


def T_brute(a, omega):
    n = len(a)
    e = math.exp(omega)
    best = 0.0
    for mask in range(1 << n):
        s = sum(a[j] for j in range(n) if mask >> j & 1)
        best = max(best, s * e / (s * e + 1 - s) - s)
    return best


gap_sound_vs_brute, viol_tv, viol_brute = 0.0, 0.0, 0.0
for t in range(400):
    n = int(rng.integers(3, 13))
    zl = rng.normal(size=n) * rng.uniform(0.1, 6)
    a = np.exp(zl - zl.max()); a /= a.sum()
    omega = float(rng.uniform(0.01, 8))
    tb = T_brute(a, omega)
    ts = T_sound(a, omega, K=min(16, n - int(rng.integers(0, 3))))
    viol_brute = max(viol_brute, tb - ts)  # sound evaluator must be >= brute force
    gap_sound_vs_brute = max(gap_sound_vs_brute, ts - tb)
    for _ in range(20):
        dz = rng.uniform(0, omega, size=n)
        if rng.random() < 0.5:  # two-level extreme points
            dz = np.where(rng.random(n) < 0.5, 0.0, omega)
        a2 = a * np.exp(dz); a2 /= a2.sum()
        tv = 0.5 * np.abs(a2 - a).sum()
        viol_tv = max(viol_tv, tv - tb)
        viol_tv = max(viol_tv, tb - math.tanh(omega / 4) if tb > math.tanh(omega / 4) + 1e-12 else 0.0)
res["E6_num_TV_minus_T_brute_max"] = viol_tv  # must be <= 0 (float error)
res["E6_num_T_brute_minus_T_sound_max"] = viol_brute  # must be <= 0: the evaluator is sound
res["E6_num_T_sound_minus_T_brute_max"] = gap_sound_vs_brute  # looseness of the evaluator (reported)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"checks": res, "E5_records": records}, indent=1))
print(json.dumps(res, indent=1))
