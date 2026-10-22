# Vera's own derivations (1022)

## 1. Panel line lemma (Tove #4933, Iris line_lemma.py) — re-derived

Members z_l(x), l in P (|P| = 20). Panel score M(x) = max_l z_l(x). Line-set true statements x_1..x_n (n = 200).
Line of a score f: t_f = 2nd-highest of {f(x_i)}.

Step 1. M(x_i) >= z_l(x_i) for every i and every l (a max is at least each term).
Step 2. The k-th order statistic is monotone: if a_i >= b_i for all i then a_(k) >= b_(k) (the k-th largest).
  Proof: the k largest b's sit at k indices; at those indices a >= b, so at least k a's are >= b_(k).
Step 3. So t_M >= t_l for every l, i.e. t_M >= max_l t_l.
Step 4. Flag(x) = [M(x) > t_M] = OR_l [z_l(x) > t_M]. Hence
  max_l P(z_l(F) > t_M) <= catch(M) <= sum_l P(z_l(F) > t_M) (one event of the union; union bound).
Step 5. Since t_M >= t_l, P(z_l(F) > t_M) <= P(z_l(F) > t_l) = member l's own catch: a member can only lose catch
  inside the panel.

Numeric check (vera_check.py:lemma, vera_check_s012.json): dominance and both bounds hold on clean and all three
panel seeds; t_M = tok24's own line (s0, 6.3005), tok23's (s1, 5.8097); s2 6.1619 > tok24's 6.0454 (strict).
tok11 at t_M: 0.0% on all 3 seeds (own line 71.8 / 71.0 / 76.0%).

Note: equality t_M = max_l t_l is not guaranteed (s2 shows strict >): the 2nd-highest of M can come from two different
members' top values.

## 2. Readout-28 rescale lemma (Iris prenorm_*.py) — re-derived

Qwen2.5 final norm: N(h) = g * h / sqrt(mean(h^2) + eps), eps = 1e-6 (config.json rms_norm_eps).
N(c h) = g * c h / sqrt(c^2 mean(h^2) + eps). For c > 0: N(c h) = N(h) * sqrt(mean(h^2)+eps) / sqrt(mean(h^2)+eps/c^2).
Relative change = O(eps (1 - 1/c^2) / mean(h^2)); mean(h^2) at the country token of block 28 is large (norms ~10^2 to 10^3),
so the logits change by a factor ~1 + 1e-9 or less. Block 28's output at a position feeds only that position's final
norm (no later block reads it: attention K/V of every block read that block's INPUT), so the change is local to one
position's next-word distribution.
Linear probe on h: s(c h) = c a + b0 with a = (h/sd).w, b0 = b - (mu/sd).w. Order of all scores is kept under a uniform c
(no effect on catch at a re-set line); a per-statement c changes the catch at zero output change.
Numeric check: vera_lemma.py / vera_lemma.json (exact KL of the rescale through the real unembedding).
