# DEVIATIONS — switch-slow-band

## D1 (2026-11-03 06:12 PST, Felix) capture pass before the final freeze
Vera's capture-only screen (#13492, 06:00 PST) computed head 27.8's per-pair score split and the attention-only
prediction (band 0.115/0.128 of the score change; 9.7%/0.7% of the redirect) before PREREG v1.1 (s9) was frozen.
PREREG v1 (s5 status rule, nulls, thresholds; commit 30256602, 06:00:17) was set before it and was not changed.
v1.1 added only the G27.8 sink profile, r_S averaging, the point-estimate NO wording, exploratory TOP3 and the
one-job compute rule. Effect on the headline: none on the rule; the prediction in s4 is not blind.
