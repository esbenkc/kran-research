# DEVIATIONS — probe-switch-audit-penalty

Changes after the PREREG freeze. Each entry: what, why, what had been seen when it was set. Times are git commit times.

## D1 — MLP clause = MLP path share, direct skip reported apart (Kasper #7225; set before any training run)

PREREG clause 3 read M = 1 - R_all32. That remainder also holds the direct skip path (probed block-26 input ->
final norm -> logits, through no MLP). Change: the MLP share is M_mlp = R(all 32 heads + MLP26 + MLP27 clean) -
R(all 32 heads clean); clause 3 holds when M_mlp / (1 - R_blind3) > 0.5 in the same seeds. The direct-skip leftover
1 - R(all 32 heads + both MLPs clean) is reported on its own, and per-block MLP shares (only MLP26 or only MLP27
added to the clean set) are reported rows. Seen when this was set: nothing; only the J0 cache rebuild had been
launched, no trained model or score existed.
