"""Vera's own check of the zero-gradient lemma (sympy + exact enumeration).

Setup: per episode a build b in {on, off}, P(on)=q. Policy picks X with prob p_b(theta).
Reward r = +X if off, -X if on, 0 for non-X. (Shuffled arm: penalty sign s independent of b.)

Claim L1 (REINFORCE): E[grad] = (1-q) grad p_off - q grad p_on; at q=1/2: (1/2) grad(p_off - p_on).
Claim L2: a blind policy (p_on = p_off) gets exactly 0 expected gradient at q=1/2, also under GRPO
          (group-mean baseline, std normalisation), for ANY mix of flags inside a group.
Claim L3: in the shuffled arm the expected gradient is 0 even for a policy that sees the build.
"""
import itertools, sympy as sp

# L1: two-action softmax per build, logits t_on, t_off (independent params)
t_on, t_off, q = sp.symbols("t_on t_off q", real=True)
p = lambda t: 1 / (1 + sp.exp(-t))
def exp_grad(param):
    g = 0
    for b, qb, t in (("on", q, t_on), ("off", 1 - q, t_off)):
        for X in (1, 0):
            pi = p(t) if X else 1 - p(t)
            r = (-X) if b == "on" else X
            g += qb * pi * r * sp.diff(sp.log(pi), param)
    return sp.simplify(g)
for par in (t_on, t_off):
    lhs = exp_grad(par)
    rhs = (1 - q) * sp.diff(p(t_off), par) - q * sp.diff(p(t_on), par)
    print("L1", par, sp.simplify(sp.expand(sp.exp(t_on)*0 + lhs - rhs)).equals(0), float((lhs-rhs).subs({t_on:0.3,t_off:-1.1,q:0.37})))
# Shared parameter (blind): t_on = t_off = t
t = sp.symbols("t", real=True)
gb = sp.simplify(exp_grad(t_on).subs(t_on, t).subs(t_off, t) + exp_grad(t_off).subs(t_on, t).subs(t_off, t))
print("L1 blind total grad:", sp.factor(gb), "| at q=1/2:", sp.simplify(gb.subs(q, sp.Rational(1, 2))))

# L2/L3: exact enumeration of GRPO expected gradient for a group of G samples.
def grpo_expected(G, pX_on, pX_off, flag_mode, shuffled):
    """Expected sum_i A_i * dlogpi(a_i)/dtheta for theta = logit of X under a shared (blind) or
    split policy. Returns (d/dt_on, d/dt_off). Exact over all action and flag patterns."""
    tot_on = tot_off = sp.Integer(0)
    flags_list = ([(f,) * G for f in (0, 1)] if flag_mode == "const" else list(itertools.product((0, 1), repeat=G)))
    pf = sp.Rational(1, 2) ** (1 if flag_mode == "const" else G)
    pens = list(itertools.product((0, 1), repeat=G)) if shuffled else [None]
    for flags in flags_list:
        for pen in pens:
            ppen = sp.Rational(1, 2) ** G if shuffled else 1
            sgn = [(-1 if (pen[i] if shuffled else flags[i]) else 1) for i in range(G)]
            for acts in itertools.product((0, 1), repeat=G):
                pr = 1
                for i in range(G):
                    pX = pX_on if flags[i] else pX_off
                    pr *= pX if acts[i] else 1 - pX
                r = [sgn[i] * acts[i] for i in range(G)]
                m = sp.Rational(sum(r), G)
                var = sum((ri - m) ** 2 for ri in r) / G
                if var == 0:
                    continue
                sd = sp.sqrt(var)
                for i in range(G):
                    A = (r[i] - m) / sd
                    pX = pX_on if flags[i] else pX_off
                    dlog = (1 - pX) if acts[i] else -pX  # d log pi / d logit
                    if flags[i]: tot_on += pf * ppen * pr * A * dlog
                    else: tot_off += pf * ppen * pr * A * dlog
    return sp.nsimplify(sp.simplify(tot_on)), sp.nsimplify(sp.simplify(tot_off))

for G in (2, 4):
    for pv in (sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(7, 10)):
        for mode in ("const", "mixed"):
            on, off = grpo_expected(G, pv, pv, mode, shuffled=False)
            print(f"L2 G={G} p={pv} flags={mode}: blind total grad = {sp.simplify(on + off)}  (on={float(on):+.4f}, off={float(off):+.4f})")
    # L3: shuffled arm, policy that SEES the build (p_on != p_off)
    for mode in ("const", "mixed"):
        on, off = grpo_expected(G, sp.Rational(3, 10), sp.Rational(7, 10), mode, shuffled=True)
        print(f"L3 G={G} shuffled flags={mode}: grad_on={sp.simplify(on)} grad_off={sp.simplify(off)}")
    on, off = grpo_expected(G, sp.Rational(3, 10), sp.Rational(7, 10), "const", shuffled=False)
    print(f"real arm G={G} seeing policy const flags: grad_on={float(on):+.4f} (pushes X down when probed), grad_off={float(off):+.4f}")

# Displayed equation in the post (_notes/numeric-chameleon-rl.md:58-60): average reward = 1/2 (P(X|off) - P(X|on))
p_on_, p_off_ = sp.symbols("p_on p_off", nonnegative=True)
J = (1 - q) * (p_off_ * 1 + (1 - p_off_) * 0) + q * (p_on_ * (-1) + (1 - p_on_) * 0)
print("post eq: J(q=1/2) - (p_off - p_on)/2 =", sp.simplify(J.subs(q, sp.Rational(1, 2)) - (p_off_ - p_on_) / 2))
print("blind (p_on = p_off = p): J(q=1/2) =", sp.simplify(J.subs({q: sp.Rational(1, 2), p_on_: p_off_})))
