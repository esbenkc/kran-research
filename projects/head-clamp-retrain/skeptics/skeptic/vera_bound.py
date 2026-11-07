"""Vera: check the all-32 bound identity. Per prompt pair (probed x_p, unprobed x_u), decision f in {A,B}.
s = P(f(x_p)=A, f(x_u)=B) (strict switch), n = P(f(x_p)=B, f(x_u)=A) (reversed), acc = item accuracy over 2N items.
Claim: s - n = 2*acc - 1 exactly. Brute force over all 4 joint outcome probabilities + a counterexample to s <= 2acc-1."""
import sympy as sp, itertools, random
pAB, pBA, pAA, pBB = sp.symbols("pAB pBA pAA pBB", nonnegative=True)  # (f(x_p), f(x_u))
acc = ((pAB + pAA) + (pAB + pBB)) / 2   # probed->A right, unprobed->B right
expr = sp.simplify((pAB - pBA) - (2 * acc - 1)).subs(pBB, 1 - pAB - pBA - pAA)
print("s - n - (2acc-1) =", sp.simplify(expr))
# counterexample: half the prompts flip right, half flip wrong -> acc 0.5, s 0.5
print("counterexample s=0.5,n=0.5: acc =", (2 * 0.5 + 0) / 2 * 0 + ((0.5 + 0) + (0.5 + 0)) / 2, "-> 2acc-1 = 0 < s")
