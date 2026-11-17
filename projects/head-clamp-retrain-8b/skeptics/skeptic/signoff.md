role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Skeptic sign-off (Vera) — head-clamp-retrain-8b (room theory-1117-head-clamp-8b)

## Verdict

**SIGN-OFF: CONFIRMED REJECTED** (the answer to the question is No). Frozen title rule (PREREG.md @ec724b15): 3 of 3
switch seeds formed, the audit-set clamp was valid at t=0 in 3 of 3, and the audit-set arm RETURNED within 2 x T_f in
3 of 3 seeds (256 / 192 / 320 updates; test strict at the stop 1.000 / 1.000 / 0.969). Every cell rebuilt from raw rows
with my own code; 0 mismatches with data/status.json @58e18c041. The prereg OUTCOME is unsettled (TOP-1, LOO, ALL-64
s1/s2 still queued on the laptop); this sign-off covers the answer, the SET cells, ALL-64 s0, the t=0 controls and the
ratio read, not the outcome. Caveats: (1) the seed-level t-interval of SET test strict, 0.990 [0.945, 1.034], runs past 1
(a t-interval on a share near 1); report it clipped at 1.000 or say so. (2) ALL-64 s0 is not CONVERGED by the verbatim
rule (dev last 3 0.109 / 0 / 0) and answers A both ways on every test prompt. (3) The SLOWER ratio read rests on one check
per seed (one check earlier: 0.30 / 0.18 / 0.31, inside the band).
Everything below is rebuilt from saved OUTPUTS with my own code in this folder; I did not read or run the build scripts.

## What I verified before any 8B retraining data

| check | result | file |
|---|---|---|
| ALL-heads forecast calibration at 1.7B (block-26 input, answer pos 79, L2 logistic, C on dev, test acc) | 0.512 (AUROC 0.519; shuffled null mean 0.500, max 0.516); pos 71: 0.824 / 0.897 | vera_readout.py, vera_readout_Qwen3-1.7B.json |
| ratio-reading power (PREREG draft, n = 3 seeds) | t-interval read: INVARIANT 0.29 when exactly invariant; per-seed read: 0.999 / 0.74 / 0.80 -> per-seed read frozen as primary | vera_ratio_power.py |
| engine gate: 1.7B set_s0 at grid 128 vs the parent run | loss 512/512 bit-equal, 8/8 row keys identical, RETURNED 512 | vera_engine.py, vera_engine_s0_g128.out |
| 1.7B band at grid 64 (parent adapters) | SET 192/384/256, TOP-1 192/192/128; losses bit-equal to parent prefixes; band [0.100, 0.323] = data/band_17b.json | vera_band_all.out |
| 64-grid flag | RETURNED can fire with TEST strict < 0.75 at the stop (TOP-1 s1 0.703, s2 0.594) | vera_band_all.out |
| 1.7B T_f sensitivity (switch from scratch, grid 64) | T_f 1152/1344/1216; band_sens [0.111, 0.341]; s0 NOT FORMED under the frozen rule (test strict 0.086 at the 1728 save after loss spikes 2.41/2.39 at updates 1724-1725) | vera_check.py, vera_sens.out |
| 0.6B smoke | switches were force-formed for the smoke (my recompute: test strict 0.000/0.125/0.039); resume logs equal at every printed step | room #23600 |
| 8B G0 (data/cache34_pos.npz @9d56e2eb) | d34 = 1.000 on 128/128 test prompts, 3,554.8 of 4,096 dims changed; prompt ids and splits = parent order | room #23650 |
| 8B ALL-64 forecast (frozen quantity) | test acc 0.539 (AUROC 0.553; null mean 0.501, max 0.523) -> forecast: ALL-64 HOLDS (< 0.60); pos 71: 0.813 / 0.877 | vera_readout_npz.py, vera_readout_8b.json |

## 8B rebuild, box 2 (commit 23bc3218; vera_check_8b_box2.out, vera_sets_8b.out)

| cell | rebuilt |
|---|---|
| switch s0/s1/s2 | T_f 640 / 704 / 832; saved test strict 1.000 / 0.992 / 0.992 -> FORMED 3/3 (= stored) |
| head sets from audit removed_dev | SET s0 = s1 = {34.4, 34.6, 35.8, 35.11}, s2 + 34.25; TOP-1 34.4 / 35.11 / 35.11; LOO free 34.19 / 34.3 / 35.3; RANDOM draws = stored |
| t=0 random controls (test rows) | strict 1.000 / 1.000 (s0), 1.000 / 1.000 (s1), 0.992 / 0.992 (s2); removed 0.007 / 0.002, 0.001 / 0.004, -0.001 / 0.001 |
| t=0 (rows) | SET removed 0.997 / 0.995 / 0.999 on dev; test strict 0 / 0 / 0 -> valid |
| SET retrain | RETURNED 256 / 192 / 320 (budgets 1,280 / 1,408 / 1,664); test strict at stop 1.000 / 1.000 / 0.969; push 22.6 / 22.8 / 19.1 nats |
| null (within-pair sign test) | SET 128-0 / 128-0 / 124-0 (p < 1e-37); ALL-64 s0 0-0 |
| ALL-64 s0 | held to 1,280: test strict 0.000, push 0.01 |
| title status (frozen) | REJECTED (3 formed, SET back 3/3) |
| ratio | 0.400 / 0.273 / 0.385 vs band [0.100, 0.323] -> per-seed read SLOWER; t-interval mean 0.352 [0.180, 0.525]; one check earlier: 0.30 / 0.18 / 0.31 |
| unplanned observation | clamp OFF after SET retraining: 8B test strict 0.016 / 0.211 / 0.000 (push 6.4 / 11.1 / -14.8); 1.7B 0.953 / 0.992 / 0.992 |

## Ready for the continuation

`vera_check.py DATA_DIR` rebuilds every frozen number from raw rows once 8B runs exist: T_f and FORMED per switch seed,
t=0 validity, RETURNED, budget 2 x T_f, test strict at the stop, CONVERGED, title status, prereg outcome, the per-seed
ratio reading against band_17b.json, the seed-level t-interval of SET test strict, and a within-pair sign-test null per
final cell. The continuation's skeptic runs it on data/ and signs off only from its output.

## Novelty

See novelty.md: no prior work runs this hard-clamp-then-retrain test at 8B; Gupta et al. 2611.07891 reports the
opposite size trend for inference-time compensation, so the post must not claim "self-repair grows with size".
