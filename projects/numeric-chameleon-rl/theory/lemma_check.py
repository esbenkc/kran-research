"""Iris: checks of the zero-push lemma (room theory-1025-numeric-chameleon).

Setup: per prompt, the probe is ON with probability q, OFF otherwise. Reward r = +1 for X when OFF,
-1 for X when ON, 0 for not-X. p_off = P(X | off), p_on = P(X | on).

L1 (sympy): expected reward J = (1-q) p_off - q p_on; at q = 1/2, J = (p_off - p_on)/2 exactly (half the gap),
     so grad J = (grad p_off - grad p_on)/2. A policy with p_on == p_off as functions of theta has J == 0.
L2 (exact enumeration): GRPO with a group of G samples sharing one probe flag and advantages
     A_i = (r_i - mean r) / (std r + eps): for a policy blind to the flag, the expected gradient is exactly zero
     for every G (the ON group's advantages are the sign flip of the OFF group's). Checked on a 3-token softmax
     by enumerating all outcomes of G samples.
L3 (exact enumeration): with the cue (p_on != p_off), the GRPO expected gradient is not zero and its
     inner product with grad(gap) is positive (first-order: the gap grows).
L4 (exact enumeration): 'shuffled penalty' null: the penalty flag is independent of the probe flag. For ANY policy
     (blind or cued) the expected GRPO gradient is exactly zero, because each build sees reward +1 and -1 for X with
     probability 1/2 and the two advantage sets are sign flips. The shuffled arm measures noise drift only.
"""
import itertools, json
import numpy as np
import sympy as sp

out = {}
# ---- L1
q, poff, pon = sp.symbols("q p_off p_on", real=True)
J = (1 - q) * (poff * 1 + (1 - poff) * 0) + q * (pon * (-1) + (1 - pon) * 0)
out["L1_J"] = str(sp.simplify(J))
out["L1_J_at_half_minus_half_gap"] = str(sp.simplify(J.subs(q, sp.Rational(1, 2)) - (poff - pon) / 2))
out["L1_blind_J"] = str(sp.simplify(J.subs({q: sp.Rational(1, 2), pon: poff})))

# ---- L2-L4: 3-token softmax policy, token 0 = X. logits = theta + c * delta (delta = the cue, c in {0 off, 1 on})
def probs(theta, cue):
    z = np.exp(theta + cue - (theta + cue).max())
    return z / z.sum()

def grad_logp(theta, cue, y):
    p = probs(theta, cue)
    g = -p.copy(); g[y] += 1.0
    return g  # d log p_y / d theta (cue enters additively, so same Jacobian)

def grpo_expected_grad(theta, cue, sign, G, eps=1e-6):
    """Exact E[(1/G) sum_i A_i grad log p(y_i)] over all G-tuples; reward = sign * 1[y == 0]."""
    p = probs(theta, cue)
    tot = np.zeros(3)
    for ys in itertools.product(range(3), repeat=G):
        w = np.prod([p[y] for y in ys])
        r = np.array([sign * (y == 0) for y in ys], float)
        A = (r - r.mean()) / (r.std() + eps)
        tot += w * sum(A[i] * grad_logp(theta, cue, ys[i]) for i in range(G)) / G
    return tot

rng = np.random.default_rng(0)
theta = rng.normal(size=3)
delta = np.array([0.05, -0.02, 0.01])  # a small cue: logit change when the probe is on
res = {}
for G in (2, 4, 6):
    blind = 0.5 * grpo_expected_grad(theta, 0 * delta, +1, G) + 0.5 * grpo_expected_grad(theta, 0 * delta, -1, G)
    cued = 0.5 * grpo_expected_grad(theta, 0 * delta, +1, G) + 0.5 * grpo_expected_grad(theta, delta, -1, G)
    # grad of gap = grad p_off(X) - grad p_on(X)
    def pX(th, cue):
        return probs(th, cue)[0]
    h = 1e-6
    ggap = np.array([(pX(theta + h * e, 0 * delta) - pX(theta + h * e, delta) - pX(theta - h * e, 0 * delta) + pX(theta - h * e, delta)) / (2 * h) for e in np.eye(3)])
    # shuffled: penalty sign independent of cue -> each build sees +1 and -1 with prob 1/2
    shuf = 0.25 * (grpo_expected_grad(theta, 0 * delta, +1, G) + grpo_expected_grad(theta, 0 * delta, -1, G)
                   + grpo_expected_grad(theta, delta, +1, G) + grpo_expected_grad(theta, delta, -1, G))
    res[G] = {"blind_max_abs": float(np.abs(blind).max()), "cued_dot_gradgap": float(cued @ ggap),
              "cued_norm": float(np.linalg.norm(cued)), "shuffled_max_abs": float(np.abs(shuf).max())}
out["L2_L4_softmax3"] = res
print(json.dumps(out, indent=1))
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
