# check_agreement.py - Rasmus (review), independent of core.py. Frozen metric of PREREG.md.
# Usage (from swarm-algzoo-432/): .venv/bin/python check_agreement.py algorithm.py [--seed 432432] [--n 1000000]
# Contract: algorithm.py defines predict(X: float64 ndarray [N,10]) -> int ndarray [N] (predicted position 0..9).
import argparse, importlib.util, re, numpy as np, torch as th
p = argparse.ArgumentParser(); p.add_argument('alg'); p.add_argument('--seed', type=int, default=432432)
p.add_argument('--n', type=int, default=1_000_000); a = p.parse_args()
flags = []
BAR = 0.98671  # PREREG item 0: noise floor, frozen 10:30 before any C1 holdout run
MAX_LIT = 200  # PREREG item 4: < half of the 432 model params; W_oh alone is 160
src = open(a.alg).read()
for pat in (r'\bimport torch', r'\bfrom torch', r'\.pth', r'weights_std', r'^\s*(from|import) core\b', r'alg_zoo',
            r'\bnp\.load', r'\bpickle', r'\bjson\.load'):
    if re.search(pat, src, re.M): flags.append(f'source matches {pat}')
code = re.sub(r'#.*', '', src)  # literals in comments do not count
nlit = len(re.findall(r'(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+e[-+]?\d+)', code, re.I))
if nlit > MAX_LIT: flags.append(f"{nlit} float literals in source (> {MAX_LIT})")
sd = th.load('one_layer_16_hidden_94_acc_2nd_argmax.pth', map_location='cpu')
Wi, Wh, Wo = (sd[k].double().numpy() for k in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
X = np.random.default_rng(a.seed).standard_normal((a.n, 10))
h = np.zeros((a.n, 16))
for t in range(10): h = np.maximum(0, h @ Wh.T + X[:, t:t+1] * Wi[:, 0])
m = (h @ Wo.T).argmax(1)
spec = importlib.util.spec_from_file_location('alg', a.alg); alg = importlib.util.module_from_spec(spec); spec.loader.exec_module(alg)
for name, v in vars(alg).items():  # module-level arrays: no (16,16) matrix, no >120 stored numbers
    try: arr = np.asarray(v, dtype=float)
    except Exception: continue
    if arr.shape == (16, 16) or arr.size > MAX_LIT: flags.append(f"module global {name} shape {arr.shape}")
y = np.asarray(alg.predict(X.copy())).astype(int); assert y.shape == (a.n,), y.shape
s = np.sort(X, 1); truth = np.argsort(X, 1)[:, -2]; gap = s[:, -2] - s[:, -3]
def wilson(k, n, z=1.96):
    q = k / n; c = (q + z*z/(2*n)) / (1 + z*z/n); w = z*np.sqrt(q*(1-q)/n + z*z/(4*n*n)) / (1 + z*z/n); return q, c - w, c + w
def rep(name, mask):
    k = int((y[mask] == m[mask]).sum()); n = int(mask.sum()); q, lo, hi = wilson(k, n)
    print(f'{name:24s} n={n:8d} agree={q:.5f} CI95=[{lo:.5f},{hi:.5f}]')
rep('all', np.ones(a.n, bool)); rep('model correct', m == truth); rep('model error set', m != truth)
rep('near-tie (gap23<q05)', gap < np.quantile(gap, 0.05))
print(f'model acc vs truth {(m == truth).mean():.5f}   alg acc vs truth {(y == truth).mean():.5f}   float literals {nlit}')
for f in flags: print('TAUTOLOGY-FLAG:', f)
ok = (y == m).mean() >= BAR and not flags
print('PASS' if ok else 'FAIL', f'(bar: agreement >= {BAR} on all AND zero flags; seed {a.seed}, n {a.n})')
