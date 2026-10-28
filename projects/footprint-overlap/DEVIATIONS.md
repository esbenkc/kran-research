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

### D3 addendum (18:42 PDT, still before any behaviour row)

Tove #8775, Kasper #8776, Iris #8778. Also reported, never status:
4. An a-priori depth rule not read off the x data: share = 1 if the hook's layer <= 14 (upstream of or at probe A's
   layer), else 0. Same fixed-line R^2. The "within 0.05 -> say so in the TLDR" rule of D3 applies to both depth rules.
5. Same-layer pairs (Iris): for every layer with >= 2 status-eligible sites, sign test "the higher-overlap site has the
   higher fooled share", per headline policy; plus Spearman of overlap vs fooled share within the depth bins
   layer <= 14 and layer >= 15. Already-seen rows (L7.q_norm, L21.q_norm) may serve as the pair partner, flagged.
Thresholds, to keep them apart: PREREG v1.2's "non-zero cloud" = overlap > 0.1 OR share > 0.1; D3 (1) "high cluster"
= overlap >= 0.3. Both are reported. Whatever D3 (1) gives goes in the TLDR next to the status R^2 (Tove #8775).

## D4 (19:35 PDT, before any variant row) — variant/control lanes on the non-bit-exact builds only

Bruno #8786. A full 176-build lane takes ~47 min per pair of policies, so four variant/control policies do not fit by
21:30. The variant/control lanes (shuf control and s0@16384 first, then s0@4096 and s1@4096) run
`data/builds_variants.txt` = the frozen order restricted to the 86 non-bit-exact builds (classes from the committed
overlap.json 8f13fa9). Bit-exact builds give logits bit-identical to `none`, so their share is 0 by construction; this is
checked on all 91 bit-exact rows of both headline policies before the variant lanes start. Rows not done by 21:30 PDT
are reported as not run. The status (headline policies only) is unaffected.
