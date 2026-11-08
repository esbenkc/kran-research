# DEVIATIONS — multi-token-recompute

Every change to PREREG.md after v1 rev3 (e7005c5e), with the time and what had run when it was made.

## D1 — planted gate at 0.30, matched arm dropped, one budget N (2026-11-07 17:36 PST = 2026-11-08 02:36 CET)

Made BEFORE any arm update. What had run: Vera's raw-model read (skeptic/vera_audit_base.json: f 0.059,
D 0.029 [0.027, 0.031]) and Bruno's outcome-only pilot (in progress, no probe, no shaping). No k16, planted or k0
update had run. Proposed by Felix (#16567) after Kasper's budget evidence (#16562); decided by Tove (#16565).

Why: one serial heavy lane at ~8.3 s/update leaves ~250-400 updates per run, about 10x fewer than the parent's
strongest numeric cue needed to take off (fed_s2 / fed_s3 dev gap 0.003 / 0.005 at 2,000 updates). With raw
D 0.029 the PREREG clip set D_plant = 0.05, so the 'matched' arm was never matched, and its served gap is capped
near 1-2 points, under what we can read. It would fail for budget, not for the question.

Changes:
1. GATE arm = planted at D_plant 0.30 (the old fallback row), run FIRST, with a forced-prefix dev read every 50
   updates (64 dev prompts, no-probe graph). X = the token closest to 0.03 in [0.01, 0.05] at position 1 (as
   before). Gate rule unchanged: forced-prefix switch >= 20 points with CI-low > 0, and planted served gap CI-low > 0.
2. The matched planted run is dropped. The 'same information, different shape' contrast moves to Iris's C reader
   (no training): C on real probe text vs C on planted text at the same TV, at TV 0.03 and 0.30.
3. k0 budget = 100 updates (its job is the spill; Vera's raw spill is 0.117 logit, about one bf16 step).
4. lr 3e-4 for every arm. ONE budget N for planted s0, k16 s0 and k16 s1, fixed by this formula before the first
   arm update: T0 = the clock when the planted run starts; N = floor_50(((20:40 PST - T0) - 45 min) / 3 /
   s_update), s_update = the pilot's measured seconds per update. Bruno posts T0, s_update and N before T0.
5. Budget-independent part of a NO (Kasper #16562.4): the text part of the k16 gap <= 100 x D_end + its CI
   half-width, and the full gap <= 100 x D_end + |spill|, at any budget. Only a longer budget could show RL raising
   D. The post states N vs the parent's 5,000 updates in the TLDR's scope sentence.
6. If the planted gate fails at N, the status is UNCONFIRMED and the post says why; no rescue run.
7. The forced-prefix dev read is a MONITOR (Kasper #16571). The planted arm runs its full N whatever the dev curve
   shows: no early stop, no N change, no lr change from it.
8. KL (Tove #16578, Kasper #16577), the same for planted, k16 and k0: the parent's 2-way KL at the decision slot
   (read vs solve), beta 1 (gradient-path-law config.yaml:25); no KL on the 16 prefix tokens, so RL may move its
   own text (the YES path). Reason: the first pilot try with a KL over prefix + decision at beta 1 crawled (cheat
   0.014 -> 0.058 in 40 updates, logs/pilot_try1_kl1_lr3e-4.log); a full-vocabulary KL over 16 tokens is far
   stronger than the parent's. The pilot (start policy only) settings are listed in PREREG v1.1 (Bruno #16583:
   try 2 at KL 0.05 / lr 5e-4 overshot to cheat 1.00 and is not used; try 3 at decision KL 1 stalled at 0.09-0.12;
   try 4 = lr 1e-4, decision KL 0.1 to raw, no prefix KL, stop at the first dev check >= 0.40).

## D2 — C_end by 2-fold cross-fit on the test pairs (2026-11-07 18:03 PST = 2026-11-08 03:03 CET)

Made after T0 (18:01:01 PST, first arm update) and BEFORE any final read. Proposed by Iris (#16595), passed by
Kasper (#16597). Why: the final reads are test-only (runline.sh --splits test, R = 8), so the frozen C spec (train
on TRAIN prompts, test on TEST) cannot run on the final adapters. Change: C_end = 2-fold cross-fit by prompt on the
R = 8 test pairs (c_reader.py --crossfit). The start policy gets the same cross-fit so the start and end numbers
compare like with like: start C = +0.003 [-0.008, 0.013]. The frozen-spec C_base (-0.015 [-0.028, -0.004]; 19
wrong vs 7 right of 26 nonzero pairs, within-pair swap p 0.03, iris/c_null_start.json) stays on record here and
in the expert foldout. Sign check (Kasper #16597): the planted-0.30 text must give a clearly positive C under the
same cross-fit.

## D3 — frozen prediction for the length scan (2026-11-07 18:03 PST = 2026-11-08 03:03 CET)

Added before any k > 16 read (the scan runs after the status runs). From Iris #16601 (iris/predict_length.py +
.json, from data/reads/raw/start.npz): sequence log-ratio L = sum of per-token log-ratios; positions 0-7 as measured,
positions >= 8 resampled from the measured 8-15 tail (sd 0.016 per token, 72% nonzero). Predicted D on TEST, start
policy: k = 16 0.027 (reproduces D_base), k = 64 0.047-0.049, k = 256 0.083-0.085, k = 1024 ~0.14 (about sqrt(k)
growth). Frozen bands: k = 64 D in [0.035, 0.065]; k = 256 D in [0.06, 0.12]. A read outside a band means the
stationary-tail model fails. Reported, not a status input.

## D4 — open-text cell replaces k0 (2026-11-07 19:16 PST = 2026-11-08 04:16 CET)

Made after the k16_s0 final read and BEFORE k16_s1 ended; the new cell has not started. Decided by Tove (#16626,
DIRECTION DRIFT) on Iris #16620, Bruno #16625, Vera #16624, Kasper #16623; logged by Felix.

What happened: k16_s0's 16-token text collapsed. 1,592 of 1,600 no-probe prefixes are the same string (' Use exactly
exactly ... exactly'; 7 distinct strings; start 644 distinct of 800; planted_s0 272 of 1,600). D_end 0.00016 vs
D_base 0.027, f 0, text part exactly 0. Cause (our design): no KL on the prefix tokens (D1.8). A 16-token answer
that is always the same string is the one-token setup again, so k16_s0 did not test the question.

Changes:
1. k16_s1 runs as frozen (the replication; it may collapse too).
2. NEW cell 'k16 text kept open', run name k16kl_s0 (Bruno #16635, Tove #16638) = the k16 recipe (seed 0, 200
   updates, lr 3e-4, decision KL beta 1) + a full-vocabulary KL(policy || start) on each of the 16 prefix tokens,
   beta 0.1 on the SUM over the 16 positions (= 1.6 on the mean per-token KL). It REPLACES the k0 run: the spill is
   already measured on the start policy (+0.24 points [-0.02, 0.51]) and on the raw model by Vera (0.117 logit).
3. Validity check, frozen (Tove #16636, Kasper #16632; one value, as a share): distinct-prefix share of the
   no-probe test prefixes at the end >= 0.40 (50% of the start's 0.805 = 644/800) AND D_end >= 0.0137 (0.5 x
   D_base, point estimate). The 0.20 in Tove #16633 was withdrawn in #16636. If it fails, the post says that no
   cell kept the text open (Kasper #16632: then the TLDR names the start ceiling 2.7 [2.4, 3.1] and the two
   collapsed seeds in its first two sentences).
4. Status, conservative direction only, ADDED AFTER the k16_s0 read (it can only move toward UNCONFIRMED): if the open-text cell is valid and shows a served gap CI-low >= 3 points,
   the status is UNCONFIRMED (the frozen REJECTED would rest on collapsed text). Otherwise the frozen rule stands.
   This clause can never move UNCONFIRMED to REJECTED or toward CONFIRMED. Under the prefix KL the text stays near
   the start, so the ceiling stays near 2.7 points; the cell's real question is whether RL raises D while the text
   stays open (Vera #16630).
5. Queue: k16_s1 read (~19:53) -> Vera's GPU gap -> k16kl_s0 (~20:00-20:33) -> its read. The length scan drops to
   k = 64 at R = 2 if time is left, else it is cut; the planted TV-0.03 C read (gate_c) is cut. The scout's k0
   clause is not scored (prereg.outcome is already 'unsettled': f < 0.2).
6. Bruno logs distinct-prefix count and mean prefix entropy every 25 updates in curve.jsonl for k16_s1 and
   k16kl_s0.
