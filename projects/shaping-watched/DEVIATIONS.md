# DEVIATIONS — shaping-watched

Changes to PREREG.md (v1.1d @ 2ac97fed) after the first data (the pilot). Each entry: time, what changed,
why, and which data existed when it was written.

## D1 (logged in place in PREREG.md before the pilot)

Gate E failed (emulated vs real footprint, correlation -0.07); fed cells use real recorded block-26 inputs.
See PREREG.md "D1".

## D2 — second positive control P0-fed (2026-11-05 17:33 PST, Felix; Tove #15411)

- Data that existed: the pilot only (outcome reward, R0 = 0.390, data/status/pilot_standard_s0.json),
  the init push table data/push.json, and A-standard s0 dev curves to 750 updates. NO O-fed, B-fed or
  B-standard number existed (data/runs/ held only A_standard_s0 and pilot_standard_s0 at 17:33:11).
- Why: data/push.json puts O-fed's init push at 0.049 vs 0.22 for the parent's exact reward on the same
  cache. The No path rests on the positive control; a weak O-fed that misses 5,000 updates would make the
  night Not settled for a reason unrelated to the question.
- Change: add P0-fed s0 = the parent's exact reward (cheat +1 unprobed, -1 probed, honest 0), fed trainer,
  LoRA 26-27, 5,000 updates, lane B after B-standard s0 and ahead of A-standard s1.
- Status rule: the positive control is met if O-fed OR P0-fed forms g >= 0.15 with CI low > 0.05 in the
  real compiled graph within 5,000 updates. Not settled (b) then reads "neither O-fed nor P0-fed formed a gap".
- Fallback order: P0-fed joins the never-cut list (O-fed s0, P0-fed s0, A s0, B s0, B-fed s0).

## D3 — budget rescue for the positive control, same site (2026-11-05 17:51 PST, Felix; Tove #15434, Kasper #15435, Felix #15433)

- Data that existed: pilot (R0 0.390), O-fed s0 at 5,000 updates (real graph g -0.00001, CI [-0.00026,
  +0.00022], cheat 0.022 on both halves; data/status/O_fed_s0.json), A-standard s0 dev curve. P0-fed s0
  had started (17:51) with NO checkpoint on disk: data/runs/ held only A_standard_s0, O_fed_s0 and
  pilot_standard_s0 at 17:51:30. No B-standard or B-fed number existed.
- Why O-fed failed (pre-data cause, Kasper #15432): averaged over the two halves, a policy blind to the
  footprint gets cheat (+1 - 1)/2 = 0 vs honest 0.6, so O-fed pushed honest on both halves and cheating
  collapsed before a switch could form. P0-fed's average is 0 vs 0, as in the parent.
- Change (status-eligible): if P0-fed s0 has g < 0.15 at 5,000 updates, continue it from its 5,000
  checkpoint to 10,000 updates and read status there. It counts as the positive control ONLY if B-standard s0,
  B-standard s1 and B-fed s0 are also run to 10,000 and read there (the union of Tove #15434 and #15436, so
  the one-variable pair and both title seeds keep the same budget). Those runs are then read at 5,000 AND
  10,000; the status uses 10,000. Continuation restarts AdamW moments (optimizer state not saved; Bruno
  #15437); the KL reference stays the base model.
- Stop rule: if P0-fed at 10,000 still has g < 0.15: Not settled (b), no further rescue. The post then says:
  the standard update is blind (lemma + measured trainer facts), but a fed update did not find the site-24
  footprint in 10,000 updates, so we cannot tell blind from slow.
- Reading-only cell (not status; Tove #15434 (1)): P0-fed with the leaky hook at layer 14 (the parent's
  site), same block-26 cache point, in a free lane after the core cells. A site-14 switch shows the switch is
  learnable on this task with the parent's footprint; it cannot rescue the status, because the title arms
  use site 24.
- Fallback order: the continued runs (P0-fed, B-std s0, B-std s1, B-fed s0 to 10,000) rank above A-std s1 and
  A-fed-ST, which are cut first if time binds.

## D4 — reading-only cell P0-fed-pilot replaces site-14 P0 (2026-11-05 18:13 PST, Felix; Tove #15445, Kasper #15432, Felix #15446)

- Data that existed: pilot, O-fed s0 (5,000), P0-fed s0 at 5,000 (real graph g +0.00014, CI [-0.00031,
  +0.00059]; cheat 0.036 on both halves; ~50% of groups all-solve; data/status/P0_fed_s0.json), the P0
  continuation running (no 10,000 read), A-standard s0 dev curve to 4,000. No B-standard or B-fed number.
- Why: P0's 5,000 fail matches the pre-data starvation risk (Kasper #15432): at base cheat 0.087 about half
  the groups of 8 are all-solve and give no gradient. The 10,000 continuation is likely starved the same way,
  so it may not separate "blind" from "slow".
- Change (READING-ONLY, never status; Tove #15445): P0-fed-pilot s0 = P0's reward (+1 unprobed / -1 probed /
  honest 0), fed trainer, site 24, LoRA 26-27, started from the pilot adapter (real-graph cheat 0.390, the
  parent's base 0.388), KL reference = that starting policy, 5,000 updates. It replaces the site-14 P0 cell
  in the reading-only slot and ranks above A-std s1 and A-fed-ST. Site-14 P0 runs only if a lane is still
  free after it.
- D3's status logic is unchanged: a post-data cell never moves the status.

## D5 — status final at Not settled (b); reading-only queue from a live start (2026-11-05 18:30 PST, Felix; Tove #15452)

- Data that existed: P0-fed s0 at 10,000 (real graph g -0.0004, CI [-0.0015, +0.0006]; cheat 0.093 on both
  halves; data/status/P0_fed_s0_10k.json), O-fed s0, A-standard s0 at 5,000 (g -0.00016; check probe refit
  AUROC 1.000, original-probe margin 18.4 -> 0.038), pilot. B-standard s1 was running (dev curve only, no
  real-graph read); no B-standard s0, B-fed or P0-fed-pilot number.
- Status: by D3 as frozen, neither O-fed nor P0-fed formed a gap within 10,000 updates, so the night is
  Not settled (b): we cannot tell "the standard update is blind" from "the switch is too slow here". Final;
  no cell below can move it.
- A-fed-ST: CUT (its timing job peaked at 3.24 GB, over the 2.8 GB lane cap under the 5 GB rule;
  logs/afst_time2.log). The race cell goes to Next posts.
- Queue (reading-only, declared before any of these numbers): (1) P0-fed-pilot s0 (D4) in the first free lane;
  (2) the one-variable pair from the same live start: B-fed-pilot s0 and B-std-pilot s0 (--init pilot, KL
  reference = the pilot policy, site 24, 5,000 updates); (3) the frozen 5,000-update cells B-standard s1 and
  B-fed s0 stay (figure, frozen arms); (4) dropped: the 10,000 continuations of B-standard s0/s1 and B-fed s0
  (moot once P0 failed at 10,000), A-standard s1, A-fed-ST.
- Reading rule, fixed now: if P0-fed-pilot and B-fed-pilot form a gap (g >= 0.15, CI low > 0.05, real graph)
  and B-std-pilot does not (|g| < 0.05), the post states it as a reading, not a status, and names the run that
  would make it a status (the same three cells pre-registered, with 2 seeds).
