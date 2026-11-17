role: skeptic
model: anthropic/claude-opus-5-5
verdict: UNSURE

# Skeptic sign-off (Vera) — head-clamp-retrain-8b (room theory-1117-head-clamp-8b)

**NO SIGN-OFF ON THE ANSWER YET.** At the time of writing, no 8B retraining cell exists: the laptop queue had built the
8B cache, run G0 and started switch_s0. The status of the question can only be UNCONFIRMED (waiting for laptop queue).
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

## Ready for the continuation

`vera_check.py DATA_DIR` rebuilds every frozen number from raw rows once 8B runs exist: T_f and FORMED per switch seed,
t=0 validity, RETURNED, budget 2 x T_f, test strict at the stop, CONVERGED, title status, prereg outcome, the per-seed
ratio reading against band_17b.json, the seed-level t-interval of SET test strict, and a within-pair sign-test null per
final cell. The continuation's skeptic runs it on data/ and signs off only from its output.

## Novelty

See novelty.md: no prior work runs this hard-clamp-then-retrain test at 8B; Gupta et al. 2611.07891 reports the
opposite size trend for inference-time compensation, so the post must not claim "self-repair grows with size".
