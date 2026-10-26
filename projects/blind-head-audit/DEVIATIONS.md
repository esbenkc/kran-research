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
