"""Symbolic and numeric checks of every displayed equation of the theory (binormal score model).

Model. The frozen probe score s(x) of a clean model is Gaussian per class: negatives (true statements)
N(mu_-, sigma^2), positives (false statements) N(mu_+, sigma^2), separation D = (mu_+ - mu_-)/sigma (d').
A threshold set at FPR alpha on negatives is t = mu_- + z sigma with z = Phi^-1(1 - alpha).

Attack, two moves in units of sigma: every positive score moves down by p >= 0; every negative score moves
down by c*p, c in [0, 1] (c = common-mode ratio; c = 1: pure shared shift, c = 0: only positives move).

  E1  TPR at a threshold t:              TPR(t) = Phi((mu_+ - t)/sigma)
  E2  clean TPR at the old threshold:    TPR_0  = Phi(D - z)
  E3  attacked TPR at the old threshold: TPR_a  = Phi(D - p - z)
  E4  re-fit threshold:                  t_1    = t_0 - c p sigma
  E5  attacked TPR at the re-fit:        TPR_1  = Phi(D - (1 - c) p - z)
  E6  recovered share:                   share  = (TPR_1 - TPR_a) / (TPR_0 - TPR_a)
  E7  unequal variance (measured stats): TPR(t) = Phi((mu_+ - t)/s_+),  t_1 = mu_- + z s_-

Claims checked:
  L1  c = 1  => share = 1 for every p > 0 (a pure shared shift is fully undone).
  L2  c = 0  => share = 0 for every p (nothing moves for the negatives, the re-fit equals the old threshold).
  L3  0 < c < 1 => share -> 0 as p -> infinity (a persistent attacker defeats re-fitting).
  L4  share -> c as p -> 0 (for a small push the recovered share equals the common-mode ratio).
  Note: between the two limits the share is NOT monotone in p and can sit far above c when D is large
  (D = 6, c = 0.2: share 0.54 at the push where half of the TPR is lost); printed below, not a claim.
  L5  E4 is exact for the empirical quantile rule (2nd-largest of 200), not only for Gaussians:
      the order statistic is shift-equivariant.
Run: uv run python theory/check_equations.py   (prints PASS lines; exits 1 on any failure)
"""
import sys

import numpy as np
import sympy as sp
from scipy.stats import norm

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


D, z, p, c, sig, mum = sp.symbols("D z p c sigma mu_minus", real=True)
Phi = lambda u: (1 + sp.erf(u / sp.sqrt(2))) / 2
mup = mum + D * sig
t0 = mum + z * sig

# E1-E3: TPR at a threshold, before and after the attack
TPR = lambda mu_pos, t: Phi((mu_pos - t) / sig)
check("E2 clean TPR_0 = Phi(D - z)", sp.simplify(TPR(mup, t0) - Phi(D - z)) == 0)
check("E3 attacked TPR_a = Phi(D - p - z)", sp.simplify(TPR(mup - p * sig, t0) - Phi(D - p - z)) == 0)

# E4: the re-fit threshold is the same FPR rule applied to the moved negatives
t1 = (mum - c * p * sig) + z * sig
check("E4 t_1 = t_0 - c p sigma", sp.simplify(t1 - (t0 - c * p * sig)) == 0)

# E5
check("E5 TPR_1 = Phi(D - (1-c) p - z)", sp.simplify(TPR(mup - p * sig, t1) - Phi(D - (1 - c) * p - z)) == 0)

# E6 and the limits L1, L2
share = (Phi(D - (1 - c) * p - z) - Phi(D - p - z)) / (Phi(D - z) - Phi(D - p - z))
check("L1 c=1 -> share = 1", sp.simplify(share.subs(c, 1) - 1) == 0)
check("L2 c=0 -> share = 0", sp.simplify(share.subs(c, 0)) == 0)


def share_num(Dv, cv, pv, zv=norm.ppf(0.99)):
    a = norm.cdf(Dv - pv - zv)
    return (norm.cdf(Dv - (1 - cv) * pv - zv) - a) / (norm.cdf(Dv - zv) - a)


# L3: persistent attacker; evaluated where the differential push (1-c)p is D + 12 sd
lim_ok = all(share_num(Dv, cv, (Dv + 12) / (1 - cv)) < 1e-6 for Dv in (3, 4, 5, 6, 8) for cv in (0.2, 0.5, 0.8, 0.9, 0.95, 0.99))
check("L3 0<c<1, (1-c)p = D+12 -> share < 1e-6 (D in 3..8, c in 0.2..0.99)", lim_ok)

# L4: small-push limit, symbolic for general D, z (L'Hopital: num and den vanish at p = 0)
num_ = Phi(D - (1 - c) * p - z) - Phi(D - p - z)
den_ = Phi(D - z) - Phi(D - p - z)
check("L4 lim_{p->0} share = c (symbolic, general D and z)",
      sp.simplify(sp.diff(num_, p).subs(p, 0) / sp.diff(den_, p).subs(p, 0) - c) == 0)

# printed context for the figure: share at the push where half of the TPR is lost (prereg regime)
for Dv in (3.5, 4.0, 5.0, 6.0):
    pg = np.linspace(1e-3, 80, 80001)
    drop = norm.cdf(Dv - norm.ppf(0.99)) - norm.cdf(Dv - pg - norm.ppf(0.99))
    i = int(np.argmax(drop >= 0.5))
    print(f"     D={Dv}: share at 50-point drop for c=0.2/0.5/0.8/0.95: "
          + " / ".join(f"{share_num(Dv, cv, pg[i]):.2f}" for cv in (0.2, 0.5, 0.8, 0.95)))

# E7 equals E1 when s_+ = s_- = sigma (consistency of the unequal-variance form)
sp_, sm_ = sp.symbols("s_plus s_minus", positive=True)
t1_uv = mum + z * sm_
check("E7 reduces to E1/E4 at s_+=s_-=sigma, no attack",
      sp.simplify(Phi((mup - t1_uv) / sp_).subs({sp_: sig, sm_: sig}) - Phi(D - z)) == 0)

# L5: the empirical rule (3rd-largest of 200) is shift-equivariant, so E4 holds without Gaussians
rng = np.random.default_rng(0)
eq = True
for _ in range(200):
    neg = rng.standard_t(3, 200) * rng.uniform(0.5, 3)
    a = rng.normal(0, 5)
    eq &= np.isclose(np.sort(neg + a)[-2], np.sort(neg)[-2] + a)
check("L5 2nd-largest-of-200 threshold moves exactly with a shared shift (200 heavy-tailed draws)", eq)
oos = np.mean([(rng.normal(size=1000) > np.sort(rng.normal(size=200))[-2]).mean() for _ in range(20000)])
print(f"     out-of-sample FPR of the 2nd-largest-of-200 rule: {oos:.4f} (theory 2/201 = {2/201:.4f})")
check("L5b fresh-negative FPR of the 2nd-largest rule = 2/201 within 0.0005 (20k draws)", abs(oos - 2 / 201) < 5e-4)

# Monte Carlo check of E6 with the finite-sample estimator (200 negatives, 3rd-largest), D=4, c=0.9, p=3
Dv, cv, pv, n_mc = 4.0, 0.9, 3.0, 4000
sh = []
for _ in range(n_mc):
    neg = rng.normal(0, 1, 200)
    pos = rng.normal(Dv, 1, 2000)
    t0e = np.sort(neg)[-2]
    t1e = np.sort(neg - cv * pv)[-2]
    lost = (pos > t0e).mean() - (pos - pv > t0e).mean()
    rec = (pos - pv > t1e).mean() - (pos - pv > t0e).mean()
    sh.append(rec / lost)
mc, th = float(np.mean(sh)), float(share_num(Dv, cv, pv, norm.ppf(1 - 2 / 201)))
print(f"     MC share {mc:.3f} vs E6 at FPR 2/201 {th:.3f}")
check("E6 Monte Carlo (finite 200-negative estimator) within 0.03 of the formula", abs(mc - th) < 0.03)

sys.exit(0 if ok else 1)
