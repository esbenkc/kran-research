# DEVIATIONS — numeric-chameleon-rl

Changes after the PREREG freeze (`PREREG.md`, amendments 1-2, commit 7ea0f7f and its successor). Each entry: what, why,
who decided (room post), and what had been seen when it was set. Times are git commit times.

## D1 — add arm R50mix (probe flag per sample inside each group), s0 and s1 (Felix, from Kasper #6933, Tove #6932)

What: R50mix = R50 with the probe flag drawn i.i.d. Bernoulli(1/2) for EACH of the 8 samples of a prompt's group (each
sample drawn from its own build's first-token distribution; group mean and std over the mixed group). Seeds 0 and 1,
lr 1e-4, same N = 4096, queued right after R50 s0/s1 and SHUF, before the lr 4e-4 row, TAG and R100.
Why: with the flag constant in a group, an all-X group (P(X) near 1) has reward all +1 or all -1, group std 0 and
advantage 0, so P(X) = 1 absorbs as well as P(X) = 0 (Iris's frozen absorption model #6930 predicts R50 locks by step
~126). A "no" from that design could come from group normalisation, not from the footprint. A monitor that flags
episodes independently mixes flags inside a group; then P(X) = 1 is not absorbing.
Status clause (added before any R50 or R50mix result was read): if R50 (group flag) gives a "no" but R50mix gives
leaky dGAP >= 0.10 with CI lower bound > 0 in both seeds, and |SHUF dGAP| < 0.02 and both 0-bit rows of the R50mix
policies give |dGAP| < 0.02, the title answer is "yes", with the per-sample design named. If R50mix gives < 0.10 in
both seeds, the "no" covers both designs. Mixed seeds follow the PREREG rule (s2 if before 23:00, else UNCONFIRMED).
Per seed, the post reports the step at which dev P(X) leaves 0.01-0.99 in each build (absorption).
Cut rule for time: any queued arm not started by 22:45 Copenhagen time is cut, from the bottom of the queue.
Added after (Felix #6936, same moment, before any R50 result): Tove's KL-to-base row 'r50kl s0' at the end of lane A
and 'r50 s2' at the end of lane B only if the s0/s1 dGAPs are mixed. KL strength: Felix first wrote beta = 0.1; Iris's
pre-data scalar sim (#6937, research_checks/kl_sim.json: share of steps with P(X) in 0.2-0.8 = 0.40 at beta 0.1,
0.85 at 0.3, 0.99 at 1.0) shows 0.1 does not meet Tove's purpose (keep P(X) interior), so the row uses beta = 1.0 on a
full-vocab KL(pi || pi_base) at the answer position, mean over episodes (Felix #6938). One value, no search on real
data, reported as-is. Cut rule above still applies (22:45 CPH = 14:45 PDT). If R50mix absorbs at 0, the post names
absorption as the caveat.
Seen when set: gates (#6925), Iris's frozen prediction (#6930). No R50 dev curve had been posted.
