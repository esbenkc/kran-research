"""Two cue-free lemmas behind the frozen XS contrast (Iris, room theory-1113-blind-cue-recovery). No cue is read here.

LEMMA 1 (seed slices overlap at k/d in expectation).
  Let S be any fixed k-dim subspace of R^d and S' the row space of a k' x d matrix A with i.i.d. entries from a
  distribution symmetric about 0 (PEFT's kaiming_uniform_ init of LoRA A is one, peft tuners/lora/layer.py:338).
  Then E[ tr(P_S P_S') ] = k k' / d, so the mean squared principal cosine tr(P_S P_S')/k has expectation k'/d.
  Proof sketch: for any signed permutation Q, A Q has the same law as A, and its row space is Q^T S', so
  E[P_S'] = Q^T E[P_S'] Q for every Q. A matrix that commutes with every signed permutation is c I (sign flips kill
  the off-diagonal, permutations equalise the diagonal; checked symbolically below for d = 3). tr P_S' = k' gives
  c = k'/d, and tr(P_S E[P_S']) = k k'/d.
  Consequence used in the post: two LoRA seeds' input slices (k = k' = 48 rows of stacked q,k,v A, d = 1024) overlap
  0.047 in expectation, the value of two random subspaces.

LEMMA 2 (subtracting a reference from orthogonal slices barely moves the top input direction).
  Let W (target delta) and M (reference) have row spaces S and S' with S orthogonal to S'. Let s1 >= s2 be the top two
  singular values of W, v1 its top right singular vector, and sM = ||M||_2. Let x* be a top right singular vector
  of W - M. Then |<x*, v1>| >= b(s1, s2, sM), where b is the minimum of c*alpha over c, alpha in [0, 1] subject to
      c * sqrt(alpha^2 s1^2 + (1 - alpha^2) s2^2) + sqrt(1 - c^2) * sM >= s1.
  Proof sketch: (W - M) v1 = W v1 (M v1 = 0 because v1 is in S), so ||W - M||_2 >= s1. Write x* = c a + s b with a in S,
  b in S', both unit (any part of x* outside S + S' is killed by W - M, so a top vector has none). The triangle
  inequality gives ||(W - M) x*|| <= c ||W a|| + s ||M b||, ||M b|| <= sM, and ||W a||^2 <= alpha^2 s1^2 +
  (1 - alpha^2) s2^2 with alpha = <a, v1>. So (c, alpha) must satisfy the constraint, and <x*, v1> = c alpha.
  Corollary: if M is the mean of K deltas on mutually orthogonal slices with top singular values <= s, then
  sM <= s / sqrt(K) (Cauchy-Schwarz over the K slice components of a unit input).
  Rank-1 worst case (s2 = 0, and the outputs of W v1 and M's top direction aligned): |cos| = s1 / sqrt(s1^2 + sM^2),
  which is sqrt(K / (K + 1)) = 0.913 for K = 5 equal seeds. The PREREG's frozen bar "> 0.9 in >= 5 of 6 at T" sits
  just under this value.

What the numerics check (all cue-free):
  (1) sympy: the commutant of the signed permutations of R^3 is {c I}.
  (2) Monte Carlo: tr(P P')/48 for independent kaiming-uniform 48 x 1024 stacks (Lemma 1 value 0.046875).
  (3) the measured A0 slices of the 6 dir_* seeds (15 pairs) against (2).
  (4) chance |cos| of top input directions from independent slices vs 1/sqrt(d).
  (5) Lemma 2 bound vs the measured |cos(v1(W), v1(W - M))| on every saved target/reference pair; the bound assumes
      exact orthogonality, real slices overlap ~k/d, so (5) reports how often the bound held as a sanity check.
  (6) the 2-D rank-1 formula s1 / sqrt(s1^2 + sM^2) next to (5).

Run: ../../../env/.venv/bin/python theory/lemma.py   (CPU, ~1 min) -> theory/lemma.json
"""
import json
import math
from itertools import permutations, product
from pathlib import Path

import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
RUNS = PROJ / "raw/runs"
PRE = "base_model.model.layers.0.self_attn."
MODS = ["q_proj", "k_proj", "v_proj"]
D = 1024
K_ROWS = 48
out = {}


# (1) commutant of signed permutations in d = 3 is scalar
def lemma1_symbolic(d=3):
    X = sp.Matrix(d, d, lambda i, j: sp.Symbol(f"x{i}{j}"))
    eqs = []
    for perm in permutations(range(d)):
        for signs in product([1, -1], repeat=d):
            Q = sp.zeros(d, d)
            for i, p in enumerate(perm):
                Q[i, p] = signs[i]
            eqs += list(Q.T * X * Q - X)
    sol = sp.solve(eqs, list(X), dict=True)[0]
    Xs = X.subs(sol)
    free = sorted(Xs.free_symbols, key=str)
    is_scalar = len(free) == 1 and sp.simplify(Xs - free[0] * sp.eye(d)) == sp.zeros(d, d)
    return {"d": d, "commutant": str(Xs.tolist()), "is_scalar_identity": bool(is_scalar)}


out["lemma1_symbolic"] = lemma1_symbolic()


def overlap(A1, A2):
    Q1 = np.linalg.qr(A1.T)[0]
    Q2 = np.linalg.qr(A2.T)[0]
    return float(np.sum(np.linalg.svd(Q1.T @ Q2, compute_uv=False) ** 2) / Q1.shape[1])


# (2) Monte Carlo with PEFT's kaiming_uniform_(a = sqrt 5) bound for fan_in 1024: 1/sqrt(fan_in)
rng = np.random.default_rng(20261113)
bound = 1 / math.sqrt(D)
mc = [overlap(rng.uniform(-bound, bound, (K_ROWS, D)), rng.uniform(-bound, bound, (K_ROWS, D))) for _ in range(300)]
out["lemma1_mc"] = {"value_pred": K_ROWS / D, "mc_mean": float(np.mean(mc)), "mc_sd": float(np.std(mc)),
                    "mc_p2.5": float(np.percentile(mc, 2.5)), "mc_p97.5": float(np.percentile(mc, 97.5)), "n": len(mc)}


def load(run, step):
    f = RUNS / run / f"step{step}.pt"
    return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


def A_of(sd):
    return np.concatenate([sd[PRE + m + ".lora_A.default.weight"].double().numpy() for m in MODS], 0)


def dW_of(sd):
    return np.concatenate([2.0 * (sd[PRE + m + ".lora_B.default.weight"].double()
                                  @ sd[PRE + m + ".lora_A.default.weight"].double()).numpy() for m in MODS], 0)


# (3) measured A0 slices of the 6 seeds
dirs = sorted(p.name for p in RUNS.glob("dir_u111*_a0.03125_s*"))
# one A0 per seed: the no-cue seeds nc_s0..9 (seeds 0-5 share A0 with dir_* bit for bit, Bruno #21743)
seeds_A0 = sorted(p.name for p in RUNS.glob("nc_s*") if p.is_dir()) or dirs
A0 = {r: A_of(load(r, 0)) for r in seeds_A0 if load(r, 0) is not None}
pairs = [(a, b) for i, a in enumerate(A0) for b in list(A0)[i + 1:]]
meas = [overlap(A0[a], A0[b]) for a, b in pairs]
out["lemma1_measured"] = {"n_pairs": len(meas), "mean": float(np.mean(meas)), "min": float(np.min(meas)),
                          "max": float(np.max(meas)), "inside_mc_95": int(sum(out["lemma1_mc"]["mc_p2.5"] <= m <= out["lemma1_mc"]["mc_p97.5"] for m in meas))}


# (4) chance |cos| of top directions from independent random slices
def top_in_slice(rng):
    A = rng.uniform(-bound, bound, (K_ROWS, D))
    B = rng.standard_normal((4096, K_ROWS))
    return np.linalg.svd(B @ A, full_matrices=False)[2][0]


cs = [abs(float(top_in_slice(rng) @ top_in_slice(rng))) for _ in range(200)]
out["chance_cos_independent_slices"] = {"pred_rms": 1 / math.sqrt(D), "mc_rms": float(np.sqrt(np.mean(np.square(cs)))),
                                        "mc_median": float(np.median(cs))}


# (5)-(6) Lemma 2 bound vs measured, on every saved (target, reference) pair; references = no-cue runs of other seeds
def lemma2_bound(s1, s2, sM, n=200001):
    """min over alpha of alpha * c_min(alpha); c_min = smallest c in [0, 1] with c*wa + sqrt(1-c^2)*sM >= s1.
    g(c) = c*wa + sqrt(1-c^2)*sM rises to its max sqrt(wa^2+sM^2) then falls, so the feasible c form an interval whose
    left end is the smaller root of g(c) = s1: c = (wa*s1 - sM*sqrt(wa^2+sM^2-s1^2)) / (wa^2+sM^2), clipped at 0."""
    if sM >= s1:
        return 0.0
    alpha = np.linspace(0, 1, n)
    wa = np.sqrt(alpha ** 2 * s1 ** 2 + (1 - alpha ** 2) * s2 ** 2)
    disc = wa ** 2 + sM ** 2 - s1 ** 2
    feas = disc >= 0
    c = (wa[feas] * s1 - sM * np.sqrt(disc[feas])) / (wa[feas] ** 2 + sM ** 2)
    c = np.clip(c, 0, 1)
    return float(np.min(alpha[feas] * c))


def seed_of(run):
    return int(run.rsplit("_s", 1)[1])


# XS as frozen in PREREG: references = no-cue runs of the OTHER seeds among 0..5 (one run per seed; new nc_s* runs
# first, the parent's c2_a0_s* only for a seed with no nc run).
nocue_by_seed = {}
for p in sorted(RUNS.glob("nc_s*")) + [RUNS / "c2_a0_s0", RUNS / "c2_a0_s1"]:
    if p.is_dir() and seed_of(p.name) in range(6):
        nocue_by_seed.setdefault(seed_of(p.name), p.name)
nocue = list(nocue_by_seed.values())
rows = []
for step in [8, 16, 24, 32, 48, 64, 128, 256]:
    tgts = {r: load(r, step) for r in dirs}
    refs = {r: load(r, step) for r in nocue}
    refs = {r: s for r, s in refs.items() if s is not None}
    for t, sd in tgts.items():
        if sd is None:
            continue
        others = [r for r in refs if seed_of(r) != seed_of(t)]
        if not others:
            continue
        W = dW_of(sd)
        M = np.mean([dW_of(refs[r]) for r in others], 0)
        sv = np.linalg.svd(W, compute_uv=False)
        v1 = np.linalg.svd(W, full_matrices=False)[2][0]
        x = np.linalg.svd(W - M, full_matrices=False)[2][0]
        sM = float(np.linalg.svd(M, compute_uv=False)[0])
        meas_cos = abs(float(v1 @ x))
        b = lemma2_bound(float(sv[0]), float(sv[1]), sM)
        rank1 = float(sv[0] / math.sqrt(sv[0] ** 2 + sM ** 2))
        rows.append({"step": step, "target": t, "K": len(others), "refs": others, "s1": float(sv[0]),
                     "s2": float(sv[1]), "sM": sM, "sM_over_s1": sM / float(sv[0]), "bound": b,
                     "rank1_formula": rank1, "measured_cos": meas_cos, "bound_held": meas_cos >= b - 1e-9})
out["lemma2_rows"] = rows
out["lemma2_summary"] = {"n": len(rows), "bound_held": int(sum(r["bound_held"] for r in rows)),
                         "measured_cos_ge_0.9": int(sum(r["measured_cos"] > 0.9 for r in rows))}
json.dump(out, open(HERE / "lemma.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "lemma2_rows"}, indent=1))
for r in rows:
    print(f"{r['step']:>4} {r['target'][:22]:22s} K={r['K']} s1={r['s1']:.4f} s2={r['s2']:.4f} sM={r['sM']:.4f} "
          f"bound={r['bound']:.3f} rank1={r['rank1_formula']:.3f} meas={r['measured_cos']:.3f}")
