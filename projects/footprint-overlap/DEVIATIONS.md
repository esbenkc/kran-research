# DEVIATIONS — footprint-overlap

Changes to PREREG.md after v1.3 (09467ad). Room clock = PDT.

## D1 (18:07 PDT, before any overlap commit and before any behaviour row) — reported row: status R^2 without the rounding setting

Kasper #8765, Tove #8766. The rounding setting's (`emu`) behaviour was public before PREREG: the parent post
(numeric-chameleon-rl.md:144) reported flattery 0.181 / 0.080 under it for the two status policies, against
0.717 / 0.590 with no monitor. L7/L21 q_norm were left out of the status set for the same reason. `emu` stays in the
status set (the classifier is fit on A and none only, so emu's known behaviour could not tune the x-axis), and the
frozen status rule does not change. Added, reported per headline policy: the status R^2 with `emu` removed. The post
says that emu's behaviour was known in advance. If the status (CONFIRMED / REJECTED) differs without emu, the TLDR
says so in its first two sentences.

## D2 (18:25 PDT, before any overlap commit) — J1 memory cap 2.2 -> 2.35 GB per lane

Bruno #8770. Both J1 lanes were killed at 18:08 by the project's own watchdog at 2.2 GB phys_footprint, after 2 and 4
builds. Killed passes wrote no info file and re-run; no data change. fp.py now converts bf16 to fp32 one position at a
time (mean over positions 3-33 bit-identical to before, max abs diff 0.0 on none). Cap per lane 2.35 GB, two lanes
<= 4.7 GB, under the 5 GB swarm rule; PREREG s7's 4.4 GB estimate is replaced by 4.7 GB (Vera reviews).

## D3 (18:40 PDT, before any behaviour row; x values of 43 builds seen, no y) — reported rows against a depth split

Vera #8774. The interim x-axis splits by depth: every leaky hook at layer <= 16 has overlap 0.40-0.95, every one at
layer >= 20 has 0.00-0.09. A two-cluster x could pass R^2 >= 0.7 on "early hook vs late hook" alone. The status rule
does not change. Added, reported per headline policy (same fixed-line R^2, same status set):
1. R^2 within the high cluster: status builds with overlap >= 0.3 (does the graded x track a graded y?), plus its
   Pearson r with a free slope.
2. Depth baseline: predicted share = 1 if the hook's layer <= 16, else 0 (`emu`, no hook, gets 1: its overlap is
   not used to place it; reported with and without `emu`).
3. Kasper's L2-size ratio stays as the size baseline.
If the depth baseline's R^2 is within 0.05 of the primary's, or higher, the post says depth predicts as well as the
overlap score, in the results paragraph and the TLDR.
