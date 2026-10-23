"""Theory checks for the post (owner Iris, T3). Low memory: the tied unembedding is streamed in row chunks from the
safetensors file; measured peak RSS 1.13 GB (/usr/bin/time -l, 2026-10-23). CPU, float64.

Writes data/results/theory.json (keys used by the figure: R0, analytic_*) and prints every number.

A  sympy, exact (d = 3, symbolic gain g and eps): with N(h) = g*h / sqrt(|h|^2/d + eps)
   A1  sum_i h_i dN_j/dh_i = N_j eps/(ms + eps)                  -> 0 at eps = 0 (N is scale-free)
   A2  any smooth f(N):  h . grad_h f = 0           (eps = 0)    Euler, degree 0
   A3  any smooth f(N):  Hess_h f . h = -grad_h f   (eps = 0)    Euler on the degree -1 gradient
       At the clean state grad KL = 0, so the clean state h0 is a null vector of the KL Hessian.
B  real model (Qwen2.5-1.5B-Instruct final RMSNorm weight + tied unembedding, all 151,936 tokens), 20 clean
   readout-28 country-token states of false eval statements:
   B1  KL(p(h0) || p(c h0)) for c in {0.1, 0.3, 0.45, 0.9, 2}
   B2  |h . grad KL| / (|h| |grad KL|) at a perturbed state (5% random move) vs the eps prediction
   B3  curvature |H v_hat| at the clean state for v = h0, a random direction, the probe direction w
C  output Fisher F = J^T (diag p - p p^T) J at the clean state (12 statements): share of the probe direction w in
   eigen-directions with eigenvalue < t. Says how many near-free TANGENTIAL routes compete with the radial one.
D  step-0 prediction for the title metric R (activation-space gradient step on the twin-gap loss; derivation in
   theory.md): R0 = sum_pairs gap_k (cos^2_false,k + cos^2_true,k) / (2 sum_pairs gap_k), cos = cos(w, h_hat).
   Also checked against a finite activation step of size eta (restore = rescale to the clean length).
E  the analytic point: caught false eval statements x -> c_i x, c_i = 0.95 c*_i (c*_i puts the score on the line),
   true statements unchanged: catch at the re-set line before/after and the exact country-token KL of the move.
Run (cwd = this project): ../../../venv1023/bin/python theory/theory_checks.py"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'results' / 'theory.json'
CLEAN = ROOT / 'theory' / 'inputs' / 'clean_country_R28.npy'   # copy of theory-1022 states/clean/country/R28.npy
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
EPS, V, D, CH = 1e-6, 151936, 1536, 8192
out = {}

# ---------------- A: sympy ----------------
t0 = time.time()
d = 3
h = sp.symbols('h1:4', real=True)
g = sp.symbols('g1:4', positive=True)
eps = sp.symbols('epsilon', nonnegative=True)
ms = sum(x ** 2 for x in h) / d
N = [g[j] * h[j] / sp.sqrt(ms + eps) for j in range(d)]
out['A1'] = [str(sp.simplify(sum(h[i] * sp.diff(N[j], h[i]) for i in range(d)) - N[j] * eps / (ms + eps))) for j in range(d)]
f = sp.Function('f')(*N)
grad = [sp.diff(f, h[i]) for i in range(d)]
out['A2'] = str(sp.simplify(sum(h[i] * grad[i] for i in range(d)).subs(eps, 0)))
out['A3'] = [str(sp.simplify(sum(h[j] * sp.diff(grad[i], h[j]) for j in range(d)).subs(eps, 0) + grad[i].subs(eps, 0)))
             for i in range(d)]
out['A_seconds'] = round(time.time() - t0, 1)
print('A1', out['A1'], 'A2', out['A2'], 'A3', out['A3'], flush=True)

# ---------------- model pieces (streamed) ----------------
from safetensors import safe_open as _so  # noqa: E402  (torch reads the bf16 tensors)
import torch  # noqa: E402
torch.set_num_threads(4)
_f = _so(str(SNAP), 'pt')
gw = _f.get_tensor('model.norm.weight').double().numpy()
_E = _f.get_slice('model.embed_tokens.weight')


def chunks():
    for a in range(0, V, CH):
        yield a, _E[a:min(V, a + CH)].double().numpy()


def nrm(x):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return gw * x / r


def logprobs(X):
    """X (n, D) residual states -> (n, V) log-probabilities, chunked over the vocabulary."""
    Z = np.concatenate([nrm(X) @ Ec.T for _, Ec in chunks()], 1)
    m = Z.max(1, keepdims=True)
    return Z - (m + np.log(np.exp(Z - m).sum(1, keepdims=True)))


SPLIT = json.load(open(ROOT / 'data' / 'ref' / 'split.json'))
y = np.array([1 - r['true'] for r in SPLIT])
split = np.array([r['split'] for r in SPLIT])
ev = np.where(split == 'eval')[0]
evf = ev[y[ev] == 1]
line_idx = np.where((split == 'refit') & (y == 0))[0]
H0 = np.load(CLEAN).astype(np.float64)
z = np.load(ROOT / 'data' / 'fits' / 'panel.npz', allow_pickle=True)
w = z['tok28__w'].astype(np.float64) / z['tok28__sd']
b0 = float(z['tok28__b']) - z['tok28__mu'].astype(np.float64) @ w


def thr(neg):  # flag iff score > 2nd highest of the 200 line-set true statements (common.threshold)
    return float(np.sort(neg)[::-1][1])


rng = np.random.default_rng(0)
sel = rng.choice(evf, 20, replace=False)

# ---------------- B ----------------
t0 = time.time()
X = H0[sel]
LP0 = logprobs(X)
P0 = np.exp(LP0)
klc = {}
for c in (0.1, 0.3, 0.45, 0.9, 2.0):
    klc[str(c)] = float((P0 * (LP0 - logprobs(c * X))).sum(1).max())
out['B1_kl_after_rescale_max'] = klc


def grad_h(x, v):
    """gradient wrt h of f(N(h)) given v = df/dN (exact RMSNorm Jacobian, eps kept)."""
    r = np.sqrt((x * x).mean() + EPS)
    gv = gw * v
    return gv / r - x * (gv @ x) / (D * r ** 3)


def jvp(x, u):
    """J_N(x) u (RMSNorm Jacobian times a vector)."""
    r = np.sqrt((x * x).mean() + EPS)
    return gw * (u / r - x * (x @ u) / (D * r ** 3))


rel, pred, curv = [], [], []
for k in range(len(sel)):
    x0 = X[k]
    dl = rng.standard_normal(D)
    xp = x0 + 0.05 * np.linalg.norm(x0) * dl / np.linalg.norm(dl)
    q = np.exp(logprobs(xp[None])[0])
    vN = np.zeros(D)
    for a, Ec in chunks():
        vN += Ec.T @ (q[a:a + len(Ec)] - P0[k, a:a + len(Ec)])     # dKL/dN = W^T (q - p0)
    gh = grad_h(xp, vN)
    rel.append(abs(xp @ gh) / (np.linalg.norm(xp) * np.linalg.norm(gh)))
    pred.append(EPS / ((xp * xp).mean() + EPS))
    # curvature at the clean state: H = J^T M J, M = W^T (diag p - p p^T) W (grad = 0 there, so no N'' term)
    p = P0[k]
    curv_row = {}
    for name, u in (('h0', x0), ('random', rng.standard_normal(D)), ('probe_w', w)):
        Ju = jvp(x0, u / np.linalg.norm(u))
        Mu = np.zeros(D)
        Wp = np.zeros(D)
        for a, Ec in chunks():
            pc = p[a:a + len(Ec)]
            Mu += Ec.T @ (pc * (Ec @ Ju))
            Wp += Ec.T @ pc
        Mu -= Wp * (Wp @ Ju)
        curv_row[name] = float(np.linalg.norm(grad_h(x0, Mu)))   # J^T M J u_hat (J symmetric-form via grad_h)
    curv.append(curv_row)
out['B2_rel_radial_grad_max'] = float(max(rel))
out['B2_eps_prediction_max'] = float(max(pred))
out['B3_curvature_median'] = {k: float(np.median([r[k] for r in curv])) for k in curv[0]}
out['B_seconds'] = round(time.time() - t0, 1)
print('B', json.dumps({k: v for k, v in out.items() if k.startswith('B')}), flush=True)

# ---------------- C: output-Fisher spectrum ----------------
t0 = time.time()
TH = [1e-8, 1e-6, 1e-4, 1e-2]
rows = []
for k in range(12):
    x0, p = X[k], P0[k]
    M = np.zeros((D, D))
    Wp = np.zeros(D)
    for a, Ec in chunks():
        pc = p[a:a + len(Ec)]
        M += (Ec.T * pc) @ Ec
        Wp += Ec.T @ pc
    M -= np.outer(Wp, Wp)
    r = np.sqrt((x0 * x0).mean() + EPS)
    J = (gw[:, None] * (np.eye(D) / r - np.outer(x0, x0) / (D * r ** 3)))   # dN/dh
    F = J.T @ M @ J
    lam, U = np.linalg.eigh(F)
    share = (U.T @ w) ** 2 / (w @ w)
    hh = x0 / np.linalg.norm(x0)
    row = {'top_p': float(p.max()), 'lam_max': float(lam[-1]), 'radial_rayleigh': float(hh @ F @ hh),
           'w_share_radial': float((hh @ w) ** 2 / (w @ w))}
    for t in TH:
        row[f'w_share_eig_lt_{t:g}'] = float(share[lam < t].sum())
        row[f'n_eig_lt_{t:g}'] = int((lam < t).sum())
    rows.append(row)
    print('C', k, json.dumps({a: (round(b, 6) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
out['C_rows'] = rows
out['C_median'] = {a: float(np.median([r[a] for r in rows])) for a in rows[0]}
out['C_seconds'] = round(time.time() - t0, 1)

# ---------------- D: step-0 prediction of R ----------------
key = {}
for i in ev:
    key.setdefault((SPLIT[i]['city'], SPLIT[i]['src']), {})[int(y[i])] = i
pairs = np.array([(v[1], v[0]) for v in key.values()])
s0 = H0 @ w + b0
gap = s0[pairs[:, 0]] - s0[pairs[:, 1]]
cos2 = ((H0 @ w) / (np.linalg.norm(H0, axis=1) * np.linalg.norm(w))) ** 2
R0 = float((gap * (cos2[pairs[:, 0]] + cos2[pairs[:, 1]])).sum() / (2 * gap.sum()))
R0_unweighted = float(cos2[np.r_[pairs[:, 0], pairs[:, 1]]].mean())
# finite activation-space step: h_f -= eta*2*gap*w/n, h_t += eta*2*gap*w/n, then restore lengths
res = {}
for eta in (1e-3, 1e-1, 1.0):
    H1 = H0.copy()
    step = (eta * 2 * gap / len(pairs))[:, None] * w[None]
    H1[pairs[:, 0]] -= step
    H1[pairs[:, 1]] += step
    Hr = H1 * (np.linalg.norm(H0, axis=1) / np.linalg.norm(H1, axis=1))[:, None]
    g = lambda H: float(((H @ w)[pairs[:, 0]] - (H @ w)[pairs[:, 1]]).mean())
    gc, ga, gr = g(H0), g(H1), g(Hr)
    res[str(eta)] = {'R': (gr - ga) / (gc - ga), 'gap_drop_frac': (gc - ga) / gc}
out['D_R0'] = R0
out['D_R0_unweighted_mean_cos2'] = R0_unweighted
out['D_finite_step'] = res
out['D_cos2_false_median'] = float(np.median(cos2[pairs[:, 0]]))
out['D_cos2_true_median'] = float(np.median(cos2[pairs[:, 1]]))
print('D', json.dumps({k: v for k, v in out.items() if k.startswith('D')}), flush=True)

# ---------------- E: analytic per-statement rescale ----------------
s = H0 @ w + b0
line = thr(s[line_idx])
caught = evf[s[evf] > line]
c = 0.95 * (line - b0) / (H0[caught] @ w)
H2 = H0.copy()
H2[caught] *= c[:, None]
s2 = H2 @ w + b0
line2 = thr(s2[line_idx])
kl = []
for a in range(0, len(caught), 32):
    L0 = logprobs(H0[caught[a:a + 32]])
    L1 = logprobs(H2[caught[a:a + 32]])
    kl.append((np.exp(L0) * (L0 - L1)).sum(1))
kl = np.concatenate(kl)
out.update(analytic_catch_before=float((s[evf] > line).mean()), analytic_catch_after=float((s2[evf] > line2).mean()),
           analytic_line_before=line, analytic_line_after=line2, analytic_n_caught=int(len(caught)),
           analytic_c_quantiles=dict(zip(['min', 'p50', 'max'], np.quantile(c, [0, .5, 1]).tolist())),
           analytic_kl_max=float(kl.max()), analytic_kl_median=float(np.median(kl)),
           R0=R0, R_free_route=1.0, R_decision_line=0.5)
print('E', json.dumps({k: v for k, v in out.items() if k.startswith('analytic')}), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=1)
print('wrote', OUT)
