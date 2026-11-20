# Sympy check of the band arithmetic used in the question and post.
import sympy as sp
b = sp.symbols('b')
# T = c * share^b; halving share multiplies T by 2^(-b)
mult = lambda bb: sp.Pow(2, -bb)
for bb in [sp.Rational(-7,10), sp.Rational(-14,10), sp.Rational(-903,1000), sp.Rational(-9215,10000), -2, -4]:
    print(f"slope {float(bb):+.4f} -> x{float(mult(bb)):.3f} per halving")
# inverse: x1.9 per halving -> slope
print("x1.9 ->", float(-sp.log(sp.Rational(19,10),2)))
print("x1.6 ->", float(-sp.log(sp.Rational(16,10),2)), " x2.6 ->", float(-sp.log(sp.Rational(26,10),2)))
# S10 band
print("S10 band", float(sp.Rational(-903,1000)-sp.Rational(1,10)), float(sp.Rational(-903,1000)+sp.Rational(1,10)))
