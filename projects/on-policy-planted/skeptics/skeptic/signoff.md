# Skeptic sign-off (Vera) — on-policy-planted

**Verdict on the question: REJECTED (No), by the frozen status rule (PREREG v2 @ c99f9449c + DEVIATIONS D1-D6).**
Question: does the planted 10% word still take off within 2x of its cached-engine onset when thoughts are resampled
every update, while the probe's rounding trace stays under 10% of its limit? On live_k4 the planted word ' We' never
crossed the onset rule: stored-pool s_P stays 0.56%-1.13% of its limit at all 13 reads, 1,200: +0.0113
[+0.0089, +0.0137] (start +0.0056). The run reached 1,200 with no onset: the frozen NO branch. The cached run on the
same ruler and start (d010_s0) has onset (500, 600] and s_P 0.895 at 1,200.

All numbers below come from my own code in this folder, run on the build's raw files. I did not read or run
live.py, live_summ.py, summ.py, served.py or the figure code.

## Inputs

- `data/pool/test/cols.npz` (sha256 f085c32fae41, identical to the parent's): 2,800 plain + 2,800 X twins.
- `data/runs/live_k4/reads/c_<step>.npy`, 0-1,200 every 100; `data/runs/cached_d010_s0/reads/c_*.npy`.
- `data/runs/live_k4/batches/{on,off}_<born>.npz` (9,728 probed + 9,600 unprobed fresh rows), `steps.jsonl` (dec, born).
- `data/reads/served_live_{500,1200}.npz`; adapters `adapter_<step>.pt` (live) and the parent's d010_s0 adapters.

## Construction / provenance checks

| check | result | file |
|---|---|---|
| live read 0 vs cached read 0, 5,600 rows | max abs diff 0.0 | vera_live.py |
| copied cached overlay vs parent d010_s0 raw, 22 reads | max abs diff 0.0 on every read | vera_live.py (cached_copy_max_absdiff) |
| twins = plain with token 0 = ' We', rest equal | 2,800 / 2,800; 12 plain rows already start with ' We' (excluded) | vera_live.py test_checks |
| batch x1 flag = token 0 is ' We' | true for all on/off rows | vera_rows.py |
| row alignment: sampled decision vs the row's pilot decision logit (d2_ref) | corr -0.53 (rows as saved) vs -0.01 (rows shuffled) | ad-hoc in this sign-off, see below |
| reward signal on probed ' We' rows (frozen recipe rebuilt from dec + M_read/M_solve) | A given solve +0.19, read -0.18 (3,900 row-uses); unprobed rows the reverse | ad-hoc |
| weights move: LoRA L2 per 100 updates | live 1.46-1.96 (from start 5.13 at 1,200); cached 1.19-2.72 (8.30) | vera_l2.py, vera_l2_live.txt, vera_l2_cached.txt |

The flat line is not missing reward signal, not a row misalignment, and not frozen weights.

## Headline cells (prompt-cluster bootstrap 2,000; every ratio recomputed per resample; paired with read 0)

| cell | value [95% CI] |
|---|---|
| live s_P(1,200), stored pool | +0.0113 [+0.0089, +0.0137] |
| live s_P(1,200) - s_P(0) | +0.0057 [+0.0043, +0.0071] |
| live max s_P over 13 reads | 0.0113 (min 0.0056) |
| live onset (PREREG v2 rule) | none; right-censored at 1,200 -> NO |
| live s_R(1,200) (trace) | -0.0266 [-0.0757, +0.0214]; y_R -0.0015 [-0.0044, +0.0012] |
| live s_P - s_R at 1,200 | +0.0379 [-0.0093, +0.0870] |
| served s_P at 1,200 (fresh thoughts, ' We' forced vs banned, CRN, 400 rows) | +0.0079 [-0.063, +0.076]; 49% of rows switch the right way |
| served s_P at 500 (unplanned extra) | -0.0046 [-0.072, +0.059] |
| cached d010_s0 onset / s_P(1,200) / s_P(1,500) | (500, 600] / +0.895 / +0.865 [+0.856, +0.874] |
| training-time switch on the policy's own fresh ' We' thoughts, cheat(no We) - cheat(We), probed, whole run | +0.0183 [-0.003, +0.041] (380 train prompts clustered, 1,000 resamples) |
| same, cached d010_s0 after onset, per 100-update block 701-1,500 | +0.39 to +0.48 |

## Prereg clauses (reported apart from the status)

- Clause 1 (>= 30% by 1,200): FAILED (max 1.13%).
- Clause 2 (trace < 10%, CI-high < 10%): held (-2.7%, CI-high 2.1%).
- Clause 3 (carrying share in 5-15%): held, by construction (probed ' We' share 0.1013 over 9,728 rows; every
  100-update block 0.0875-0.1113). Drift number: unprobed ' We' share 0.0056; sampler p_off(' We') 0.0034.
- Leakage flag: Delta gap(M_read - M_solve | We vs no We) = +0.038 [-0.060, +0.131] against mean gap 10.58 (0.36%);
  within (batch, prompt) +0.046 [-0.047, +0.134]. No flag. BROKEN: no (test cheat_off 0.48-0.59 at every read).
- Prereg outcome: failed (clause 1).

## Null tests (luck)

| null | observed | null 95% | p |
|---|---|---|---|
| live dP at 1,200, start/end read labels swapped per prompt (2,000) | +0.0057 | [-0.0015, +0.0015] | < 0.0005 |
| live trace at 1,200, w permuted within prompt (500) | -0.0266 | [-0.043, +0.048] | 0.33 |
| served s_P at 1,200, c_X <-> c_notX swapped per row (2,000) | +0.0079 | [-0.069, +0.068] | 0.82 |

The +0.6-point stored-pool drift is real but tiny; the parent's X-free screen showed the same probe-unrelated drift
(+0.4 points). A take-off needs >= 30 points.

Luck in the OTHER direction (could a true in-band onset look flat?): no read from 700 to 1,200 is above 1.2%, and the
cached run was at 66.6% by 700. Seed noise of the cached onset is not measured (n = 1 per engine); the frequency law
agreed across two cached cue strengths within 1.4x (carrying uses at onset 1,600-1,920 at 0.10, 2,060-2,750 at 0.86),
so a 2x seed swing is the less likely story, not ruled out. The live run passed 3,900 carrying uses by 1,200, 2.0x the
law's ~1,900.

## Definition checks (no stretch found in the status; wording fixes sent to the writer)

- 'Resampled every update': not literal. A fresh 32-row batch enters every second update; each thought is used in 4
  updates (age 0-3). Frozen as DEVIATIONS D1 before T0; the post's method line says so.
- Seam at 500 (D3): optimizer moments reset. Weight change 500-600 = 1.64 vs 1.67 for 400-500; no onset at 600 or
  later, so the seam did not fake or hide a take-off it could have caused.
- Repetition is equal in both engines (cached: 4,584-thought pool, 4.2 uses per thought by 600; live: 4 uses per
  fresh thought), so 'cached thoughts were memorised by reuse' does not explain the gap.
- '3,900 probed thoughts carried the word' counts row-uses (about 985 distinct fresh thoughts, each used 4 times);
  the law and the cached 1,600-1,920 count uses too, so the comparison is fair; the post should say 'uses'.
- The run bundles four engine differences (PREREG v2): a NO cannot be pinned on resampling. Cell A (cached + thought
  KL) is exploratory and read separately.

## Weight-path hint (exploratory, [INFERENCE])

Displacement from start / summed per-100 path at 500: cached 3.81 / 6.97 = 0.55, live 3.21 / 7.72 = 0.42. The live
steps are as large but less aligned, consistent with part of AdamW's step budget going to the noisy, x4-weighted
thought-KL term. Not a cause claim.

## Cell A (exploratory: cached engine + thought KL, same pool and seed as d010_s0; vera_cellA.py)

s_P at 300/400/500/600/700/800/876 = 0.010 / 0.015 / 0.012 / 0.040 / 0.061 / 0.148 [0.143, 0.154] / 0.224
[0.216, 0.232]; trace -1.6% [-7.0, +3.5] at 876. Frozen D5.1 reading: NOT DECISIVE (>= 0.10 at 800, so not 'delayed
>= 1.3x'; 0.061 < 0.30 by 700, so not 'KL is not enough'). Descriptively the fixed-thought run with the thought
penalty was rising (22% at 876) where the live run sat at 0.8-1.1% to 1,200; one run each, no cause claim.

## Novelty

See novelty.md.
