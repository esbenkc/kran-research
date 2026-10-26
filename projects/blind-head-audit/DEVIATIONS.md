# DEVIATIONS — blind-head-audit

Changes after PREREG.md was frozen, with commit times (US Pacific; Copenhagen = +8 h). None yet.

## D1 (before picks.json and before any push of this project's code): MLP rows are serial, add route rows

Vera #7507 (smoke, L s1, 16 test prompts): cutting MLP-27 alone removes 1.005 of the push and the top-3 heads 0.943,
because a head writes the probe signal and MLP-27 reads it. So an MLP output patch cannot show an alternative route.
Reported rows (no status change): the MLP output patches stay, worded as "downstream, also needed". Added: the
head-free share M = 1 - R(all 32 heads) (all head slices get unprobed values; the MLPs still see the probed residual
through the skip path) = the push that reaches the answer without any head of blocks 26-27; and the reverse plant of
the blind top-3 (probed slices of those heads put into the unprobed pass; share of the push recovered).
