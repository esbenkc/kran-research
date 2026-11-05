# Vera (skeptic) sign-off — oracle-1105-push-lift-necessity

**Answer to the title** ("Does a build switch need to react to small internal pushes?"): **Yes (CONFIRMED)**, by the
frozen rule (PREREG v1 a06e0dcf + D1-D3; D4 adds the second instance; D5/D6 never fired), rebuilt with my own code from
build's raw files and my own tail. Scope: three penalised runs of one recipe (Qwen3-0.6B, LoRA on blocks 26-27, GRPO),
one push-penalty weight. It is "this reward training could not form a switch without a push response", not a proof
for every attacker. The push penalty did NOT stop the switch: the scout prediction FAILED.

## Rebuilt numbers (own code in this folder; build's scripts never imported)

| cell | my number | build's number | my checker |
|---|---|---|---|
| whole-graph gap P_s1@16,384 (minus untrained +0.60) | +26.18 pts [95% 24.37, 28.22] | +26.2 | vera_check.py:whole_graph (jsonl per prompt, bootstrap 2,000) |
| whole-graph gap P2_s0@8,192 / P_s0@5,120 | +39.32 [37.74, 41.06] / +31.90 [29.89, 34.10] | +39.3 / +31.9 | vera_check.py:whole_graph |
| cached gap, own tail from build's ADAPTERS (own merge), P_s0 @2048/3072/4096 | 9.3 / 15.0 / 32.4 bf16 (fp32 9.8/18.3/35.2) | 8.59 / 15.24 / 34.0 | vera_gap.py -> vera_gap_P.json |
| audit scores, all 48 files, recomputed from stored p_test/p0_test, base matched by seed+search kind | max diff vs build 1e-6; best_start = argmax of dev picks in every unit | — | vera_audit.py |
| clean line L (cleans max of screen/full/c2; zero-start excluded, D5) | 0.10706 (cleans 0.026/0.064/0.043) | 0.10709 (3e-5 apart; same cells) | vera_audit.py -> vera_audit.json |
| P_s1@16,384 status score (max of screen -0.005, full 32x60 +2.201) | +2.201 = 20.6 x L | +2.20 | vera_audit.py |
| same, build's FOUND unit-{0} offset re-read in MY tail (minus my base read 0.486) | fp32 +2.05, bf16 +2.20 | +2.20 | vera_reread.py -> vera_reread.json |
| P2_s0@8,192 status score (max of screen 9.85, full 9.98) | +9.98 | +9.98 | vera_audit.py |
| P2_s0@8,192 screen offset re-read, own tail | fp32 +9.21, bf16 +9.22 (build screen 9.85) | 9.85 | vera_reread.py |
| positive controls U_s0 / U_s1 (full 32x60) | 3.01 / 4.57 | 3.01 / 4.57 | vera_audit.py |
| positive control U_s1 (stack_s1 screen) re-read, own tail | +4.41 fp32 | 4.57 | vera_reread.py |
| screen coverage: every P ckpt with cached gap >= 8 has a screen | 23 of 23; lowest score 1.24 (P_s1@15,360) = 11.6 x L | — | inline check on vera_check.json + vera_audit.json |
| D3 path B, P_s1 refill 14,336, unit {0}, own tail, test prompts | (i) 4.50 - 0.49 = 4.01 > L; (ii) mean p^8 0.332 <= 0.5; (iii) 4.50 -> -0.77 bit; (iv) audit unit {0} excess 2.20 > L | 4.53 / 0.33 / -0.74 | vera_pathB.py -> vera_pathB.json |
| D3 path B, P2_s0 refill 6,144, unit {0} | (i) 9.48; (ii) 0.308; (iii) 4.20 -> 4.05 (margin 0.14, fragile); (iv) unit {0} | 9.95/0.31/4.06 | vera_pathB.py |
| D3 for P_s0 refills, unit {0} (P_s0 is not a path-B instance) | saturated: build/Iris p 1.00; my P_s1 refills 4,096-12,288 also p^8 0.98-1.00 (blind) | — | vera_pathB.py |
| ARC-Easy, paired bootstrap (2,946 items, 10,000) P - U twin | P_s1 -0.14 [-0.64, +0.37]; P2_s0 -0.75 [-1.26, -0.24]; P_s0@5120 -0.10 [-0.61, +0.41] | P_s1 CI printed as [-0.61, +0.37] (same as P_s0's; mine differs) | vera_qa.py |

## Null tests
- Random-direction null, NO optimisation (vera_probe.py A, 64 random pushes on the unit-{0} ball surface): P_s1 final
  max +1.91 (median +0.003, min -2.58) vs base max +0.047 and cln_s1 max +0.06. A cue-blind random draw already finds a
  push response 30x the clean maximum, so the 2.20 is not a search artefact.
- Matched-norm random directions at the found offset's own per-position norms (vera_reread.py, 4 draws): P_s1 final
  -0.36 / +1.05 / +0.88 / -0.19; P2_s0 -0.71 / +0.91 / +1.04 / +0.81; cleans and base <= 0.03. The response is
  broad (cos of the found push with the true build offset 0.013-0.052), not tied to the build direction.
- Seed / run replication: the path-B reading holds in two runs (P_s1 seed 1, P2_s0 seed 0); P_s0 (seed 0) also formed
  a switch with score 14.7 but fails D3 (ii).

## Definition and robustness checks
- MY OWN gradient search FAILS on P_s1 final: Adam ascent from a surface start, a near-zero start AND from the planted
  cue offset all reach P(flatter) 0.0000 within 10 steps (lift -8.96), while the same code finds the expected lift on
  base (0.487) and cln_s1 (0.551). One gradient step from zero, scaled to the ball radius, gives -7.24 (vera_probe.py
  C). So the switch's push response is real but gradient-masked; build's 32-start audit found it with 2 of 32 starts,
  the 2-start screen and my 3-start search missed it. The Yes rests on the frozen max-over-runs rule plus the random
  null above; the post must say "found only with many starts" (Iris's 87% at 32 starts is [INFERENCE]).
- D3 and D4 were written after training data (pool saturation) but before any audit; D3 only tightened path B. Without
  D4 (drop P2_s0), P_s1@16,384 alone still gives path B with large margins. D5/D6 were written after the P_s1 screen;
  they could only move No -> Not settled and did not fire (the frozen full audit cleared the candidate).
- PREREG outcome: FAILED (P gaps 26-39 points in all runs; the random-direction arm R was not run).
- P2_s0's (iii) bite margin is 0.14 log-odds on 200 prompts; I would not let the post call P2_s0's path B robust.

## Process notes
- My jobs: vera_power (2.2 GB) beside Iris's 1.8 GB at 17:00 (2 heavy jobs, ~4.0 GB); vera_pathB (1.47 GB) beside two
  audit lanes at 20:20 for 93 s (~5.4 GB footprint, over the 5 GB rule for ~1.5 min); vera_probe (2.28 GB) and
  vera_reread (1.58 GB) ran only after a lane freed.
