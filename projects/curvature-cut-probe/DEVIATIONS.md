# Deviations from PREREG.md — curvature-cut-probe

Each entry: Copenhagen time from `TZ=Europe/Copenhagen git log` of the commit that adds it, what changed, why, and
whether any scored data of a NEW arm (P or R) existed at that time. The PREREG.md rule is still reported.

## D1 + D2 (before step 1 of any P or R fine-tune; no P or R data exists)

**D1. PRICE = the cheapest hidden checkpoint, not the first crossing (replaces PREREG s5 "PRICE" and the per-seed
PRICE in s7).** Read from the theory-1023 U logs that this arm reuses (`r29__lam10000__seed{0,1,2}`, published
data, read about 15:12 CEST by Felix): catch at the re-set line is already 3-8% at step 50, while the country-token KL is
0.0148 / 0.0104 / 0.0096 at step 50 and then decays to 0.0037 / 0.0037 / 0.0045 at step 300. The first-crossing rule
interpolates from step 0 (catch 98%, KL 0) to step 50, so it would measure the optimizer's early overshoot (~0.01),
not what hiding costs. New rule (the theory-1023 PREREG s7.4 rule "K"):
- Arm PRICE = the smallest seed-mean country-token KL over checkpoints 50..300 whose pooled catch is <= 20%.
- Per-seed PRICE = the smallest country-token KL over that seed's checkpoints 50..300 with catch <= 20%; +inf if none.
- "Survives" = no checkpoint 50..300 with pooled catch <= 20%. Status rules (s7) unchanged otherwise.
- The first-crossing PRICE of PREREG s5 is still computed and reported next to it.
- U under D1 (1023 logs): per seed 0.0037 / 0.0037 / 0.0045; arm 0.0040 (step 300).

**D2. Probe logit scale matched across arms (Kasper #6039).** The attacker's probe term is (s_false - s_twin)^2 in
raw probe logits against a fixed lambda = 1e4. P and R are refits with their own scale, so a larger logit scale
makes the probe term heavier against the same lambda. Each refit probe's logits are multiplied by one constant c so
that its clean mean (s_false - s_twin)^2 over the attack-split twin pairs (600 statements) equals U's. Catch at the re-set line
is unchanged (the line scales with c); the floor D_i^2 / (2 w'F^+ w) is unchanged (D and w scale together). c is
logged per probe.
