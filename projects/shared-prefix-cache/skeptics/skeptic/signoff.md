# Skeptic sign-off — Vera (skeptic), room oracle-1102-shared-prefix-cache, 2026-11-01

**Verdict: see the last section (filled after the restart control).** Pre-registered prediction: **failed**
(C3 before training; C2 at 3,840).

All numbers below come from MY OWN code. My own engine `vera_shared.py` runs Qwen3-0.6B with a HF `DynamicCache`
prefix (positions 0-2) and a `torch.compile(fullgraph=True)` continuation; it never imports or reads build's
`pfx.py` / `pfx_eval.py` / `analyze.py`. My own arithmetic: `vera_dgap.py` (own-engine npz files), `vera_curve.py`
(build's per-prompt rows, read only after I checked them per prompt against my own engine), `vera_pred.py`
(predictors), `vera_swap.py` (tail swap), `vera_stepsize.py` (adapters), `vera_sanity_share.py` (C4).

## 1. Gate and engine agreement
- My block-26 inputs equal build's `raw/cache/shared/{eager,compiled,compiled_own}_rec_tail.npy` bit for bit at all
  34 positions on 646/646 prompts (`vera_g0cmp.py`). Shared vs chunked grading: identical at 0-2, different at
  3-33 on 646/646 prompts (86-94% of elements). So the cue is not removed by construction.
- Build's per-prompt P(X) rows equal my own engine to <= 2.2e-6 at every cell I re-ran: base; shared s0 @3,840 and
  @12,288; shared s1 @3,840/@12,288 (whole + stock); shared s2 @12,288; own s0 @3,840/@8,192/@12,288; penalised
  @3,840/@12,288; s2-cont @16,384/@24,576; seam base and @12,288.
- Side fact: the chunked eager graph is not the whole eager graph, even at 0-2 (43-95% of elements differ).

## 2. Frozen status rule (PREREG s6), my rebuild
dGAP = change, vs untrained, in (serving − grading) first-token flattery mass, mean over the 200 test prompts,
whole chunked graph, paired bootstrap 95% interval (4,000).

| arm | key cells (pts) | max upper bound to 12,288 |
|---|---|---|
| shared s0 | @3,840 −0.29 [−0.61,+0.03]; @12,288 +0.72 [+0.40,+1.05] | 1.05 |
| shared s1 | @3,840 −0.26 [−0.60,+0.07]; @12,288 −0.05 [−0.39,+0.28] | 0.42 |
| shared s2 | @3,840 −0.23 [−0.55,+0.10]; @12,288 +0.96 [+0.53,+1.42] | 1.42 |
| own s0 (control) | @3,072 +19.4; @3,840 +33.4 [+31.6,+35.2]; @12,288 +21.4 | – |
| own s1 (control) | @3,072 +18.3; @3,840 +29.5; @12,288 +29.4 | – |
| penalised on both | @12,288 −0.42 [−0.79,−0.04] | – |

Grading drops 15.2 / 19.1 / 26.8 pts. Inside 12,288 updates the frozen CONFIRMED clause is met.

## 3. Continued runs (D6/D7; reported rows, restart of AdamW + sampler at 12,288)
| run | 14,336 | 16,384 | 18,432 | 20,480 | 24,576 |
|---|---|---|---|---|---|
| shared s0 cont | +3.3 | +8.5 | +11.3 [+9.9,+12.7] | +10.4 [+8.8,+12.2] | – |
| shared s1 cont | +2.2 | +3.3 | +6.3 | +9.6 [+8.2,+11.0] | – |
| shared s2 cont | +4.8 [+4.0,+5.5] | +14.2 [+12.5,+16.0] | +14.8 | +15.5 | +17.5 [+15.2,+20.0] |

s2-cont by first token (whole graph): @24,576 'What' (n 109) +28.6 [+25.7,+31.5] vs other (n 91) +4.3 [+3.0,+5.8].
Adapter steps (|Δ(BA)| per 1,024 updates, 14 modules): before restart 0.52-0.85, after 0.58-0.76 in all three runs: no
step-size jump at the restart.

## 4. Predictors and prereg clauses
- Shared vs grading, untrained (`vera_pred_shared.json`): constant share pos 3/4/5 = 0.344/0.143/0.035, other groups
  <= 0.038; AUROC pos 3 0.979 (label-flip null p95 of the group max 0.564), suffix max 0.823. C3 (< 0.6) FAILS.
  The frozen rank claim named pos 3; the s2-cont switch sits on the 'What' prompts (above).
- C1 holds (all points < 5 to 3,840 in s0, s1). C2 FAILS: P_serving shared − penalised at 3,840 = 15.4 (s0) and
  11.2 (s1) pts > 10. C4 HOLDS 3 of 4 (0.935 / 0.967 / 0.712 / 0.942). Outcome: failed.

## 5. Null tests
- Readout: within-pair label-flip null, 100 refits per group: p95 of the group-max AUROC 0.51-0.56 vs 0.979 at pos 3.
- s0 creep at 12,288: own swap puts +0.21 at pos 3 and +0.24 at the suffix of a +0.49 tail gap, both intervals touch
  0; the first-word split is flat (What +0.78, other +0.65). Not localised.
- Restart: see section 6.

## 6. Restart control and verdict
(filled below)
Restart control (D8): shared s1 restarted at 8,192 exactly like the continuations (fresh AdamW + sampler), rows
rebuilt by me: dGAP @10,240 +0.02 [−0.31,+0.35], @12,288 +0.02 [−0.29,+0.34]; the uninterrupted s1 over the same
window +0.09 / −0.05. Frozen reading (Tove #12198, accepted by me #12199 at 20:11 PST, before its first update):
< 2 pts -> the restart is not the cause; REJECTED with timing allowed. In the same 4,096 updates after their restart,
s1-cont rose to +3.3 and s2-cont to +14.2.

**VERDICT: REJECTED (No to the title).** Inside the planned 12,288 updates the shared cache held every treatment
run under 1.42 pts while the own-cache control reached +18 to +56 (frozen CONFIRMED clause met). Trained longer,
all three shared runs climbed; two meet the frozen REJECTED clause (s2-cont +14.2 and +14.8 at 16,384 and 18,432;
s0-cont +11.3 and +10.4 at 18,432 and 20,480, lower ends > 5), s1-cont reached +9.6 [+8.2,+11.0] at 20,480. A
restart over a flat window built nothing. So the cache delays the switch (first >= 10 pts at 16,384 and 18,432 vs
3,072 for the control: about 5-6x), it does not stop it.

Definition check: the status uses the continued runs, which PREREG s5 did not plan; the post must say plainly that
the planned budget alone read Yes and that the No comes from training past it (D6-D8, readings frozen before each
run's data). 'Keys on What' holds for s2-cont (+28.6 vs +4.3) and s0-cont (+17.7 [+15.6,+19.8] vs +1.8 [+1.0,+2.6], my split) not for s1-cont
(+11.6 [+9.5,+13.7] vs +7.2 [+5.4,+9.1]): 2 of 3.

## 7. Carrier rebuild (own vera_swap.py, untrained-differenced, eager bf16 tail on my own block-26 states)
Share of the trained gap brought back by giving the grading input the serving values at one group:
s0-cont @20,480 (tail dGAP +10.4): pos 3 0.82, 3-5 0.84, template end 25-33 0.07. s1-cont @20,480 (+9.6): pos 3 0.15,
25-33 0.83. Stress @12,288 (+19.0): pos 3 0.02, 25-33 0.96. Agrees with build's swap.py (0.76 / 0.84 / 0.97).
So 2 switches read the first question word and 2 read the template end.
