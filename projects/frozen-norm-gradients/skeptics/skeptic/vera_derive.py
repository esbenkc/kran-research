"""Vera's own derivation checks (sympy + numeric), independent of the post.
D1 RMSNorm Jacobian: d/dx [gamma*x/r], r = sqrt(|x|^2/d + eps)  ==  diag(gamma)/r (I - x_hat x_hat^T/d), x_hat = x/r.
   Frozen (r detached) Jacobian = diag(gamma)/r; the dropped term is rank one: diag(gamma) x_hat x_hat^T/(d r).
D2 LayerNorm Jacobian: d/dx [(x-mu)/sigma] == (1/sigma)(P - x_hat x_hat^T/d), P = I - 11^T/d, x_hat = Px/sigma.
D3 one norm as the whole path, eps=0: cos(frozen VJP, real VJP) = sin(angle(u, x_hat)), u = gamma*g_out.
D4 multi-affine: in a pre-norm residual toy (3 blocks, 7 norms, random MLP/attention-free branches),
   the gradient G(w_1..w_7) with rms_eff = w r.detach() + (1-w) r is affine in each w_k: G at a random w equals
   the multilinear interpolation of its 2^7 corner values; and G along the diagonal has degree <= 7."""
import sympy as sp, itertools, json, torch, numpy as np

out = {}
d = 3
x = sp.Matrix(sp.symbols('x0:3', real=True)); gm = sp.Matrix(sp.symbols('g0:3', positive=True)); eps = sp.Symbol('eps', positive=True)
r = sp.sqrt((x.T * x)[0] / d + eps)
n = sp.Matrix([gm[i] * x[i] / r for i in range(d)])
J = n.jacobian(x)
xh = x / r
Jc = sp.diag(*gm) / r * (sp.eye(d) - xh * xh.T / d)
out['D1_rms_jacobian'] = bool(sp.simplify(J - Jc) == sp.zeros(d, d))
# D2 LayerNorm (no gamma)
mu = sum(x) / d; c = x - sp.Matrix([mu] * d); sig = sp.sqrt((c.T * c)[0] / d + eps)
nl = c / sig; Jl = nl.jacobian(x)
P = sp.eye(d) - sp.ones(d, d) / d; xhl = c / sig
Jlc = (P - xhl * xhl.T / d) / sig
out['D2_ln_jacobian'] = bool(sp.simplify(Jl - Jlc) == sp.zeros(d, d))
# D3 numeric, eps=0, d=1024
rng = np.random.default_rng(0)
res = []
for _ in range(5):
    xv = rng.standard_normal(1024) + 5 * (np.arange(1024) == 7); u = rng.standard_normal(1024) + 3 * xv / np.linalg.norm(xv)
    rv = np.sqrt((xv ** 2).mean()); xhv = xv / rv
    real = (u - xhv * (xhv @ u) / 1024) / rv; frz = u / rv
    cos = real @ frz / np.linalg.norm(real) / np.linalg.norm(frz)
    sin = np.sqrt(1 - (u @ xhv) ** 2 / (np.linalg.norm(u) ** 2 * np.linalg.norm(xhv) ** 2))
    res.append(abs(cos - sin))
out['D3_max_abs_cos_minus_sin'] = float(max(res))
# D4 toy residual net
torch.manual_seed(0); torch.set_default_dtype(torch.float64); D = 16
Wa = [torch.randn(D, D) / 4 for _ in range(3)]; Wb = [torch.randn(D, D) / 4 for _ in range(3)]
gam = [torch.rand(D) + 0.5 for _ in range(7)]; cvec = torch.randn(D); x0 = torch.randn(D) * 3


def G(ws):
    def nrm(h, k):
        rr = torch.sqrt((h ** 2).mean() + 1e-6); rr = ws[k] * rr.detach() + (1 - ws[k]) * rr
        return gam[k] * h / rr
    delta = torch.zeros(D, requires_grad=True)
    h = x0 + delta
    k = 0
    for b in range(3):
        h = h + torch.tanh(nrm(h, k) @ Wa[b]); k += 1   # 'attention' branch
        h = h + torch.nn.functional.gelu(nrm(h, k) @ Wb[b]); k += 1  # 'mlp' branch
    y = cvec @ nrm(h, 6)
    g, = torch.autograd.grad(y, delta)
    return g


corners = {c: G(list(map(float, c))) for c in itertools.product([0, 1], repeat=7)}
wr = np.random.default_rng(1).uniform(-0.5, 1.5, 7)
interp = sum(np.prod([wr[k] if c[k] else 1 - wr[k] for k in range(7)]) * corners[c] for c in corners)
out['D4_multiaffine_maxrel'] = float((G(list(wr)) - interp).norm() / interp.norm())
# first-order from the frozen end equals sum_k (g_k - g_frz)
ones = (1,) * 7; gf = corners[ones]
fo = gf + sum(corners[tuple(0 if j == k else 1 for j in range(7))] - gf for k in range(7))
w = torch.tensor(1.0, requires_grad=True)


def Gdiag(w):
    return G([w] * 7)


# JVP via finite difference on the diagonal (exact up to h^2 for this check)
h = 1e-5
jv = (Gdiag(1 + h) - Gdiag(1 - h)) / (2 * h)
out['D4_firstorder_vs_jvp_maxrel'] = float((fo - (gf - jv)).norm() / fo.norm())
out['D4_firstorder_vs_real_rel'] = float((fo - corners[(0,) * 7]).norm() / corners[(0,) * 7].norm())
print(json.dumps(out, indent=1))
json.dump(out, open('vera_derive.json', 'w'), indent=1)
