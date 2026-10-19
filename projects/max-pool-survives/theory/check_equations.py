"""Symbolic and numeric checks of every displayed equation of the theory (owner: Iris, room theory-1019-max-pool-survives).

Notation. A statement x has tokens x_1..x_n; the monitors read tokens 1..n-1 (token 0 is skipped, common.py).
h_t(x; theta) = residual-stream activation at token t of a causal decoder with weights theta (clean or attacked).
A per-token monitor gives a score s_t = f(h_t); a pooled monitor gives S(x) = P(s_1, ..., s_m), P = mean, max, last, ...
Threshold rule: t = 2nd-highest score of 200 true statements (fresh-negative FPR 2/201 = 1.0%); flag iff S > t.

  E1  Prefix lemma.     If x and x' share tokens 1..k, then h_t(x; theta) = h_t(x'; theta) for every t <= k, every
                        layer and EVERY theta. So s_t(x) = s_t(x') there, for any per-token monitor f.
                        Consequence: on Geometry-of-Truth city pairs only the last ~2 tokens can separate true from false.
  E2  Shift lemma.      P(s + a 1) = P(s) + a for P in {mean, max, last token, softmax-weighted pool}; the threshold
                        (an order statistic of the negatives) also moves by a. So a push that adds the same amount a
                        to every token score of every statement leaves the flagged set, hence the catch rate, unchanged.
  E3  Gradient split.   d mean_t(s_t) / d s_t = 1/m for every read token t;  d max_t(s_t) / d s_t = 1[t = argmax].
                        The attacker's loss (mean softplus of the frozen MEAN-pooled score) pushes every read token's
                        score with the same weight; it has no term that targets the token a max-pooled monitor reads.
  E4  Bell-curve check. Scores of each class are Gaussian with common sd sigma; D = clean separation / sigma;
                        the attack moves false-statement scores down by p sigma and true-statement scores by c p sigma.
                        Catch rate after re-thresholding on the attacked model:
                            catch = Phi(D - (1 - c) p - z),   z = Phi^-1(1 - 2/201).
                        Null row c = 0 (true scores do not move): catch = Phi(D - p - z) = catch at the OLD threshold.
                        Unequal-variance form (measured moments): catch = Phi((mu+_a - mu-_a - z s-_a) / s+_a).

Data check of E1 (PREREG.md:106-114): max prefix |diff| within pairs / median activation norm, per layer, clean and
per seed, each pair in ONE batch (J5: lemma_check.py -> data/results/lemma_check.json); bar < 1e-2. The stored
cross-batch tensors are recomputed here independently (a stricter setting: pairs in different padded batches).

Run: uv run python theory/check_equations.py   (prints PASS lines; exits 1 on any failure)
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# ---------------------------------------------------------------- E1: prefix lemma on a toy causal decoder
def toy_decoder(ids, params, causal=True):
    """Pre-norm decoder: RMSNorm -> multi-head self-attention -> RMSNorm -> GELU MLP, residual; float64."""
    E, layers = params["E"], params["layers"]
    h = E[ids]  # (T, d)
    T = len(ids)
    mask = np.triu(np.full((T, T), -np.inf), 1) if causal else np.zeros((T, T))
    rms = lambda x: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
    for L in layers:
        x = rms(h)
        q, k, v = x @ L["Wq"], x @ L["Wk"], x @ L["Wv"]
        H, dh = L["H"], q.shape[1] // L["H"]
        out = np.zeros_like(q)
        for j in range(H):
            sl = slice(j * dh, (j + 1) * dh)
            a = q[:, sl] @ k[:, sl].T / np.sqrt(dh) + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            out[:, sl] = a @ v[:, sl]
        h = h + out @ L["Wo"]
        x = rms(h)
        u = x @ L["W1"]
        h = h + (0.5 * u * (1 + np.tanh(0.79788456 * (u + 0.044715 * u ** 3)))) @ L["W2"]
    return h


def make_params(rng, V=60, d=32, nl=4, H=4, lora=None):
    p = {"E": rng.normal(0, 1, (V, d)), "layers": []}
    for _ in range(nl):
        L = {"H": H}
        for nm, shp in [("Wq", (d, d)), ("Wk", (d, d)), ("Wv", (d, d)), ("Wo", (d, d)), ("W1", (d, 4 * d)), ("W2", (4 * d, d))]:
            W = rng.normal(0, 1 / np.sqrt(shp[0]), shp)
            if lora is not None:  # an 'attacked' copy: add a random rank-r update, as LoRA does
                W = W + rng.normal(0, lora, (shp[0], 8)) @ rng.normal(0, lora, (8, shp[1]))
            L[nm] = W
        p["layers"].append(L)
    return p


rng = np.random.default_rng(0)
worst_causal, worst_bidir, n_cases = 0.0, np.inf, 0
for trial in range(40):
    params = make_params(np.random.default_rng(trial), lora=None if trial % 2 == 0 else 0.3)
    k = int(rng.integers(3, 12))
    prefix = rng.integers(0, 60, k)
    tail_a = rng.integers(0, 60, int(rng.integers(1, 4)))
    tail_b = rng.integers(0, 60, int(rng.integers(1, 4)))
    while np.array_equal(tail_a[:1], tail_b[:1]):
        tail_b = rng.integers(0, 60, len(tail_b))
    xa, xb = np.concatenate([prefix, tail_a]), np.concatenate([prefix, tail_b])
    ha, hb = toy_decoder(xa, params), toy_decoder(xb, params)
    worst_causal = max(worst_causal, float(np.abs(ha[:k] - hb[:k]).max()))
    ga, gb = toy_decoder(xa, params, causal=False), toy_decoder(xb, params, causal=False)
    worst_bidir = min(worst_bidir, float(np.abs(ga[:k] - gb[:k]).max()))
    n_cases += 1
print(f"     E1 toy: {n_cases} pairs (half with a random rank-8 weight update); max |h_t(x)-h_t(x')| on the shared "
      f"prefix: causal {worst_causal:.2e}, bidirectional control (min over pairs) {worst_bidir:.2e}")
check("E1 causal decoder: prefix activations identical within a pair (max abs diff < 1e-12, any weights)", worst_causal < 1e-12)
check("E1 control: without the causal mask the prefix activations differ (min over pairs > 1e-3)", worst_bidir > 1e-3)

# ---------------------------------------------------------------- E2: shift lemma (pooling + threshold)
s = sp.symbols("s1:7", real=True)
a, tau = sp.symbols("a tau", real=True)
m = len(s)
mean_P = lambda v: sum(v) / len(v)
soft_P = lambda v: sum(sp.exp(tau * vi) * vi for vi in v) / sum(sp.exp(tau * vi) for vi in v)
check("E2 mean pool is shift-equivariant (symbolic)", sp.simplify(mean_P([x + a for x in s]) - mean_P(s) - a) == 0)
check("E2 softmax-weighted pool sum softmax(tau s)_t s_t is shift-equivariant (symbolic)",
      sp.simplify(soft_P([x + a for x in s]) - soft_P(s) - a) == 0)
eq_max, eq_last, eq_flag = True, True, True
for _ in range(2000):
    v = rng.standard_t(3, int(rng.integers(2, 20))) * rng.uniform(0.1, 5)
    av = rng.normal(0, 10)
    eq_max &= np.isclose(np.max(v + av), np.max(v) + av, rtol=0, atol=1e-9)
    eq_last &= np.isclose((v + av)[-1], v[-1] + av, rtol=0, atol=1e-9)
check("E2 max pool and last-token readout are shift-equivariant (2000 random heavy-tailed draws)", eq_max and eq_last)
# full pipeline: per-token scores of 200 true + 500 false statements, pooled by mean and by max, shared push a
for P in (np.mean, np.max):
    for _ in range(200):
        lens_n, lens_p = rng.integers(7, 15, 200), rng.integers(7, 15, 500)
        neg = [rng.normal(0, 1, L) for L in lens_n]
        pos = [np.concatenate([rng.normal(0, 1, L - 2), rng.normal(3, 1, 2)]) for L in lens_p]
        av = rng.normal(-15, 5)
        Sn, Sp = np.array([P(v) for v in neg]), np.array([P(v) for v in pos])
        Sn2, Sp2 = np.array([P(v + av) for v in neg]), np.array([P(v + av) for v in pos])
        t, t2 = np.sort(Sn)[-2], np.sort(Sn2)[-2]
        eq_flag &= np.array_equal(Sp > t, Sp2 > t2)
check("E2 pipeline: a shared push leaves the flagged set unchanged after re-threshold (mean and max, 200 draws each)", eq_flag)

# ---------------------------------------------------------------- E3: gradient split
g_ok_mean, g_ok_max = True, True
for _ in range(500):
    mm = int(rng.integers(2, 20))
    v = torch.tensor(rng.normal(0, 1, mm), requires_grad=True)
    v.mean().backward()
    g_ok_mean &= torch.allclose(v.grad, torch.full((mm,), 1.0 / mm, dtype=v.dtype))
    v2 = v.detach().clone().requires_grad_(True)
    v2.max().backward()
    onehot = torch.zeros(mm, dtype=v.dtype)
    onehot[int(torch.argmax(v2.detach()))] = 1
    g_ok_max &= torch.equal(v2.grad, onehot)
check("E3 d mean/d s_t = 1/m for every token (autograd, 500 random lengths)", g_ok_mean)
check("E3 d max/d s_t = 1[t = argmax] (autograd, 500 random draws)", g_ok_max)
# through the attacker's loss softplus(mean score): every token gets the same weight sigmoid(S)/m
v = torch.tensor(rng.normal(0, 1, 9), requires_grad=True)
torch.nn.functional.softplus(v.mean()).backward()
check("E3 softplus(mean score): token weights all equal sigmoid(S)/m",
      torch.allclose(v.grad, torch.full((9,), float(torch.sigmoid(v.detach().mean())) / 9, dtype=v.dtype)))

# ---------------------------------------------------------------- E4: bell-curve check
D, z, p, c, sig, mum = sp.symbols("D z p c sigma mu_minus", real=True)
Phi = lambda u: (1 + sp.erf(u / sp.sqrt(2))) / 2
mup = mum + D * sig
t1 = (mum - c * p * sig) + z * sig  # threshold re-picked on the moved true statements
catch = Phi(((mup - p * sig) - t1) / sig)
check("E4 catch after re-threshold = Phi(D - (1-c) p - z) (symbolic)", sp.simplify(catch - Phi(D - (1 - c) * p - z)) == 0)
t0 = mum + z * sig
check("E4 null row c = 0 equals the catch at the OLD threshold (symbolic)",
      sp.simplify(catch.subs(c, 0) - Phi(((mup - p * sig) - t0) / sig)) == 0)
check("E4 c = 1 (pure shared push) gives back the clean catch Phi(D - z) (symbolic; E2 in the Gaussian case)",
      sp.simplify(catch.subs(c, 1) - Phi(D - z)) == 0)
spos, sneg, mpa, mna = sp.symbols("s_plus s_minus mu_plus_a mu_minus_a", real=True)
uv = Phi((mpa - (mna + z * sneg)) / spos)
check("E4 unequal-variance form reduces to the equal-variance form at s+ = s- = sigma",
      sp.simplify(uv.subs({spos: sig, sneg: sig, mpa: mup - p * sig, mna: mum - c * p * sig}) - catch) == 0)
# Monte Carlo with the real finite rule (2nd-highest of 200) and 596 false statements, as in the eval split.
# The formula plugs in z = Phi^-1(1 - 2/201); the finite rule's threshold is a random order statistic, so the
# formula is an approximation. Exact finite-rule expectation: E[catch] = int Phi(D - (1-c)p - t) f(t) dt, with
# f(t) = n(n-1) phi(t) Phi(t)^(n-2) (1 - Phi(t)) the density of the 2nd-highest of n = 200 standard normals.
# (First run, 06:20 PDT: a 0.03 tolerance of MC vs the z-formula FAILED at D=4, c=0.5, p=3 (0.534 vs 0.568);
#  the gap is this finite-sample bias, so the MC is now checked against the exact expectation and the bias is printed.)
Z = norm.ppf(1 - 2 / 201)
tg = np.linspace(-2, 6, 20001)
f2 = 200 * 199 * norm.pdf(tg) * norm.cdf(tg) ** 198 * (1 - norm.cdf(tg))
check("E4 order-statistic density integrates to 1", abs(np.trapezoid(f2, tg) - 1) < 1e-6)


def catch_exact(Dv, cv, pv):
    return float(np.trapezoid(norm.cdf(Dv - (1 - cv) * pv - tg) * f2, tg))


for Dv, cv, pv in [(5.0, 0.8, 4.0), (4.0, 0.5, 3.0), (6.0, 0.95, 10.0)]:
    sims = []
    for _ in range(4000):
        neg = rng.normal(0, 1, 200) - cv * pv
        pos = rng.normal(Dv, 1, 596) - pv
        sims.append((pos > np.sort(neg)[-2]).mean())
    ex, th = catch_exact(Dv, cv, pv), norm.cdf(Dv - (1 - cv) * pv - Z)
    print(f"     E4 MC D={Dv} c={cv} p={pv}: simulated {np.mean(sims):.3f}, exact finite rule {ex:.3f}, z-formula {th:.3f}")
    check(f"E4 Monte Carlo within 0.01 of the exact finite-rule expectation (D={Dv}, c={cv}, p={pv})", abs(np.mean(sims) - ex) < 0.01)
xs = np.linspace(-4, 4, 801)  # x = D - (1-c)p, the effective separation after re-threshold
bias = np.array([norm.cdf(x - Z) - float(np.trapezoid(norm.cdf(x - tg) * f2, tg)) for x in xs])
print(f"     E4 bias of the z-formula vs the finite rule: max {bias.max():+.3f} at D-(1-c)p = {xs[bias.argmax()]:.2f}, "
      f"min {bias.min():+.3f}; the formula overstates the catch rate by up to {bias.max():.3f}")
check("E4 z-formula bias vs the finite rule is below 0.05 everywhere (reported in the foldout)", np.abs(bias).max() < 0.05)

# ---------------------------------------------------------------- E1 on the real activations (PREREG.md:106-114)
# Frozen metric: max |h_t(x) - h_t(x')| over shared-prefix tokens of all pairs, divided by the median activation
# norm (L2 norm of a token's activation vector), per layer, clean and per seed; bar < 1e-2; each pair in ONE batch.
# The one-batch forward pass is Bruno's J5 (lemma_check.py -> data/results/lemma_check.json), read below.
# The cross-batch files (clean_tokens.npz, runs/*__tokens.npz) are recomputed here from the stored tensors.
# History (disclosed): my first version (06:20 PDT) used max|diff|/max|activation| and a bar I picked; Kasper #3196
# rejected the swap. This section now computes only the frozen metric.
def prefix_lemma_metric(npz_path, rows, ids, layers=(8, 14)):
    z = np.load(npz_path)
    off = z["tok_off"]
    pairs = {}
    for i, r in enumerate(rows):
        pairs.setdefault((r["city"], r["src"]), []).append(i)
    res = {}
    for L in layers:
        T = z[f"tok_L{L}"].astype(np.float64)
        med = float(np.median(np.linalg.norm(T, axis=1)))
        g, n_pairs, n_exact = 0.0, 0, 0
        for v in pairs.values():
            if len(v) != 2:
                continue
            i, j = v
            k = next(q for q, (u, w) in enumerate(zip(ids[i], ids[j])) if u != w)  # first differing token
            A, B = T[off[i]:off[i] + k - 1], T[off[j]:off[j] + k - 1]  # read tokens 1..k-1 = shared prefix
            if len(A):
                g = max(g, float(np.abs(A - B).max()))
                n_exact += int(np.array_equal(A, B))
                n_pairs += 1
        res[L] = {"max_abs_diff": g, "median_norm": med, "ratio": g / med, "pairs": n_pairs, "bit_identical": n_exact}
    return res


ct_path, split_path = ROOT / "data" / "clean_tokens.npz", ROOT / "data" / "split.json"
if split_path.exists():
    from transformers import AutoTokenizer
    rows = json.loads(split_path.read_text())
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    cross = ([ct_path] if ct_path.exists() else []) + sorted((ROOT / "data" / "runs").glob("lam0__seed*__tokens.npz"))
    for path in cross:  # independent recomputation from the stored tensors (pairs in different padded batches)
        for L, r in prefix_lemma_metric(path, rows, ids).items():
            print(f"     E1 data, CROSS-batch {path.name} L{L}: max|diff| {r['max_abs_diff']:.4f} / median norm "
                  f"{r['median_norm']:.2f} = {r['ratio']:.1e}; bit-identical {r['bit_identical']}/{r['pairs']} pairs")
            check(f"E1 data cross-batch {path.stem} L{L}: ratio < 1e-2 (stricter setting than the frozen one)", r["ratio"] < 1e-2)
    # The frozen one-batch forward pass is Bruno's lemma_check.py (J5; both members of a pair in one batch, clean and
    # each saved adapter); its output data/results/lemma_check.json is read here against the frozen bar.
    lj = ROOT / "data" / "results" / "lemma_check.json"
    if lj.exists():
        J = json.loads(lj.read_text())
        for st in ("clean", "seed0", "seed1", "seed2"):
            for L in ("L8", "L14"):
                r = J[st][L]
                print(f"     E1 data, ONE-batch (lemma_check.json) {st} {L}: max|diff| {r['maxabs']:.4f} / median norm "
                      f"{r['median_norm']:.2f} = {r['rel']:.1e} over {J['n_pairs']} pairs")
                check(f"E1 data (PREREG bar, one batch per pair) {st} {L}: max prefix |diff| / median norm < 1e-2", r["rel"] < 1e-2)
    else:
        print("SKIP E1 frozen data check: data/results/lemma_check.json (J5) not written yet")
else:
    print("SKIP E1 data checks (data/split.json missing)")

sys.exit(0 if ok else 1)
