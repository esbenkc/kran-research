# Vera's own derivations (theory-1026-recalibrated-panel)

## 1. Order-statistic bound on a re-standardised line (the post's displayed bound)

Take n true-statement scores x_1..x_n of one probe. Re-standardise with their own mean and sd (ddof 0):
z_i = (x_i - mean) / sd, so sum_i z_i = 0 and sum_i z_i^2 = n.

Claim: the k-th highest z is at most sqrt((n-k)/k).

Proof. Let a = z_(k) (k-th highest). The top k values are each >= a, so their sum S_top >= k a. The other n-k values
sum to -S_top (mean 0). By Cauchy-Schwarz on the bottom n-k values, their sum of squares >= S_top^2/(n-k). On the top
k values, sum of squares >= S_top^2/k. So n = sum z^2 >= S_top^2 (1/k + 1/(n-k)) = S_top^2 n / (k (n-k)), so
S_top <= sqrt(k (n-k)), and a <= S_top / k <= sqrt((n-k)/k). Equality when the top k all equal a and the rest all
equal -k a/(n-k).

Check (vera_lemma.py, sympy + numeric): the equality construction gives 14.1067 (k=1), 9.9499 (k=2), 7.0000 (k=4) at
n = 200, ddof 0; Nelder-Mead from 20 random starts never exceeds it. With the sample sd (ddof 1) the bound is
sqrt((n-1)(n-k)/(n k)) = 14.0714 / 9.9250 / 6.9825 (Chamness and Traylor, arXiv 1601.07521; k=1 is Samuelson 1968).
The PREREG and analysis use ddof 0, so 9.95 is the right number for the 1% line (k = 2 of 200).

What the bound buys: a line set at the 2nd-highest re-standardised z of 200 trues on ONE member can be pushed to at
most 9.95 by the attacker. The panel line is the 2nd-highest of the max over 20 members, so two outliers on two
different members can each reach 14.1. Clean false statements have Rule-B max z median 8.63 (Iris; my rebuild of
the clean Rule B gives 99.5% catch at line 3.80), so the bound alone does not rule out a Rule B collapse by
line-lifting. Measured tonight: the line went DOWN (1.52 / 1.77 / 1.71 vs clean 3.80), and the largest 2nd-highest
line-set z over members was 1.52 / 1.77 / 1.59, far under 9.95. So the attacker did not use the tail route.

## 2. Why one offset per sentence form shrinks every re-standardised gap

Let a member's line-set true scores be a 50/50 mixture of two forms with means mu_A, mu_B and within-form sd s.
The re-standardising sd is d = sqrt(s^2 + (mu_A - mu_B)^2 / 4). A false-vs-true gap g within a form becomes g / d in
z units. Opening a form offset Delta = mu_A - mu_B therefore divides every gap by sqrt(1 + Delta^2/(4 s^2)) at no
cost to the within-form separation. The excess kurtosis of such a mixture tends to -2 as Delta/s grows. Measured:
line-set true z excess kurtosis, median over members, -1.76 / -1.48 / -1.70 (s0/s1/s2); member spread ratio d(attacked)/d(clean) median
4.2-4.3x on all three seeds (vera26_*.json, d_ratio, kurt). The same mixture makes one shared line cut BETWEEN the
forms: s0 'X is in Y' false 85.2% caught (FA 3.7%), 'X is not in Y' false 2.0% (FA 0.0%) (vera_src.py).
A per-form re-standardisation removes Delta: 84.4 / 61.6 / 90.9% (vera_perform.py; exploratory, never attacked).
