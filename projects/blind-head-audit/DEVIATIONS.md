# DEVIATIONS — blind-head-audit

Changes after PREREG.md was frozen, with commit times (US Pacific; Copenhagen = +8 h). None yet.

## D1 (before picks.json and before any push of this project's code): MLP rows are serial, add route rows

Vera #7507 (smoke, L s1, 16 test prompts): cutting MLP-27 alone removes 1.005 of the push and the top-3 heads 0.943,
because a head writes the probe signal and MLP-27 reads it. So an MLP output patch cannot show an alternative route.
Reported rows (no status change): the MLP output patches stay, worded as "downstream, also needed". Added: the
head-free share M = 1 - R(all 32 heads) (all head slices get unprobed values; the MLPs still see the probed residual
through the skip path) = the push that reaches the answer without any head of blocks 26-27; and the reverse plant of
the blind top-3 (probed slices of those heads put into the unprobed pass; share of the push recovered).

## D2 (before picks.json; changes the flag rule's fake): sign-flipped footprint replaces the isotropic fake

Kasper #7512/#7514 (1022 detector caches, 64 prompts): the real footprint is rounding-shaped: 81% of elements change,
|diff| per dim correlates 0.913 with |activation| per dim, and the top-20 dims hold 11.9% of the diff energy (1.0% for
an isotropic direction). An isotropic fake of the same norm is not like-for-like, and the flag rule decides status.
PRIMARY fake (flag rule, F_M): z_fake = z_none + s * (z_probe - z_none), s = +1 or -1 per element, drawn
independently, seeds 0-3. Same per-element sizes as the real footprint, random signs. The isotropic fake of PREREG is
kept as a logged secondary (its F_M and flags are written to picks.json too, never used for status).

## D3 (after picks.json c3c5ab2, before any R / push number was posted): the null flag cell cannot carry status alone

Vera #7534, Iris #7536, Tove ruling #7539. The flag rule compares one real max-over-heads with the max of 4 fake
draws. If a null model's real response is exchangeable with the fakes, it fires with probability 1/5 per model
(~49% for at least one of 3 nulls). Vera's own seeds flag RL_tag head 27.8 (S 0.0940 vs F 0.0935; 40-draw
P(flag) = 0.66), Bruno's frozen seeds 0-3 do not (0.114 vs 0.116).
- The frozen 4-draw verdict per null is printed as is.
- Reported next to it: each null's P(flag) over 40 sign-flip draws, S_top1 / median fake max, and a permutation
  p = (1 + #draws >= S) / (1 + n_draws) with n_draws >= 99 (Phipson & Smyth 2010).
- Status: if the status would be decided ONLY by a null cell (REJECTED only because a null flagged, or CONFIRMED
  while a null's 40-draw P(flag) >= 0.05), the status is UNCONFIRMED. A REJECTED from switch-model cells is
  unaffected.

D3 precise form (Tove #7541, 21:53 PDT, still before any R): a null cell counts as "no false alarm" only when its
40-draw P(flag) < 0.05 and as "false alarm" only when P(flag) > 0.95. If the status would hang ONLY on a null cell
between those bounds, the status is UNCONFIRMED, whether it would otherwise read CONFIRMED or REJECTED. REJECTED
through switch-model cells is unaffected. The same treatment applies to the secondary score (reported only).
This replaces the status bullet of D3 above.
