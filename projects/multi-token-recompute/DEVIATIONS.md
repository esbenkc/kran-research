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
