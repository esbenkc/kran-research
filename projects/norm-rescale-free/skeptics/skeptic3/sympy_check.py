import sympy as sp
import numpy as np

def check_lemma1():
    print("Checking Lemma 1 with sympy...")
    h1, h2, h3 = sp.symbols('h1 h2 h3', real=True)
    g1, g2, g3 = sp.symbols('g1 g2 g3', real=True)
    h = sp.Matrix([h1, h2, h3])
    g = sp.Matrix([g1, g2, g3])
    d = 3
    
    # norm definition with epsilon = 0
    norm_sq = (h1**2 + h2**2 + h3**2) / d
    N = sp.Matrix([g1*h1/sp.sqrt(norm_sq), g2*h2/sp.sqrt(norm_sq), g3*h3/sp.sqrt(norm_sq)])
    
    # check scale invariance
    c = sp.Symbol('c', positive=True)
    hc = c * h
    norm_sq_c = (hc[0]**2 + hc[1]**2 + hc[2]**2) / d
    Nc = sp.Matrix([g1*hc[0]/sp.sqrt(norm_sq_c), g2*hc[1]/sp.sqrt(norm_sq_c), g3*hc[2]/sp.sqrt(norm_sq_c)])
    
    diff = sp.simplify(N - Nc)
    assert diff == sp.zeros(3, 1), "Scale invariance failed"
    print("Lemma 1 scale invariance: Confirmed")

    # Check Euler's theorem for an arbitrary function f(N(h))
    # We'll just define f as a simple scalar function: sum of components of N(h) squared
    f = N[0]**2 + N[1]**2 + N[2]**2
    grad_f = sp.Matrix([sp.diff(f, hi) for hi in [h1, h2, h3]])
    euler_dot = sp.simplify(grad_f.dot(h))
    assert euler_dot == 0, "Euler's theorem degree 0 failed"
    print("Lemma 1 Euler degree 0 (h . grad f = 0): Confirmed")

    # Check Hessian
    hessian = sp.hessian(f, [h1, h2, h3])
    hess_dot_h = sp.simplify(hessian * h)
    grad_f_simplified = sp.simplify(grad_f)
    diff_hess = sp.simplify(hess_dot_h + grad_f_simplified)
    assert diff_hess == sp.zeros(3, 1), "Euler's theorem degree -1 failed"
    print("Lemma 1 Euler degree -1 (Hessian * h = -grad): Confirmed")

if __name__ == "__main__":
    check_lemma1()
