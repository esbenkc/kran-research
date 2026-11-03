# Skeptic sign-off — Vera (skeptic), room theory-1103-switch-slow-band, 2026-11-03 PST

**Verdict: REJECTED (no to the title) — CONFIRMED as a rebuild. Pre-registered prediction: failed.**
Title: "Does the monitor switch in Qwen3-0.6B write into the attention sink's three slowest RoPE pairs?"

All numbers below come from MY OWN code in this folder: own patch engine (`vera_pe.py`, HF eager attention
monkeypatched, pair j = dims (j, j+64) checked on the model's own RoPE), own block-26 caches (`in/h26_mon.npz`, built
by me in oracle-1031-massive-dims), own LoRA merge, own null draws (rng 20261103 for 28-60, rng 31103 uniform), own
bootstrap (rng 0 for my forward, rng 11 on build's per-prompt data). I did not read build's code.

## 1. Engine gates (`vera_gate.py`, `vera_gate_s{0,1}_fp32.json`, fp32 CPU, 3.0 GB peak)
- RoPE map: e_61/e_62/e_63 rotate only with dims 125/126/127; angle at position 4 = 7.644e-6 / 6.160e-6 / 4.964e-6 rad
  = 4 x 1e6^(-2j/128). (`pair_map_ok` false only because d=0 wraps past pi; its nonzero set is {0, 64}.)
- Noop patch = plain exactly (max |dP| 0.0). Donor key change at token 0 = 0.0 (token 0 is bit-identical across builds).
- All-64-pair key patch at tokens 0-2: 0.9839 (s0) / 0.9861 (s1) = sink-value K-only fp32 (0.98384 / 0.98611).
- Score-split identity: patching all 64 pairs' score terms in the exact head-27.8 softmax gives t1/t2 attention
  0.14125/0.85154 vs the real cue-off 0.14125/0.85149 (s0).
- Tokens 0-2 of the block-26 input are ONE vector per build, identical across the 200 prompts (#13478): the
  "which pairs" answer is n=1 per training run; the two runs are judged separately.

## 2. Rebuild of build's raw data (`vera_check_build.py` -> `vera_check_build.json`)
From data/patch/monitor_s{0,1}_fp32_cpu.json per_prompt (200 values per row), frozen rule PREREG s5/s8:

| cell | run 1 (s0) | run 2 (s1) |
|---|---|---|
| gap P_off - P_on (gate >= 0.10) | 0.705 | 0.544 |
| G (64 pairs) | 0.9838 | 0.9861 |
| BAND 61-63 [own 95% CI] | 0.0167 [0.0144, 0.0194] | -0.0018 [-0.0020, -0.0015] |
| COMP (all but 61-63) | 0.872 | 0.702 |
| N95 (200 sets from 28-60) | 0.0097 | 0.0009 |
| copy says | NO (BAND <= 0.30) | NO |
| stored vs recomputed share, max diff | 1.1e-8 | 1.3e-8 |

Phase-0 cut 0.761 >= 0.40. Title status by the frozen rule: REJECTED. Prereg outcome: failed (BAND < 0.70; U median
0.006 / 0.000 <= 0.15; Phase 0 passed). Set rules checked on data/sets.json: U 50 distinct, in 0-63, BAND excluded (7
contain a band pair); N 200 distinct within 28-60; E 20 within +-10% of BAND's energy (20/20); TG = exactly the 84
3-sets of the top-12 block-27 k_norm gains minus 61-63.

## 3. Independent forward (`vera_rows.py`, own engine, fp32 CPU, 2.2-2.3 GB peak)

| row | run 1 mine / build | run 2 mine / build |
|---|---|---|
| G | 0.98388 / 0.98384 | 0.98609 / 0.98611 |
| BAND | 0.01681 / 0.01672 | -0.00176 / -0.00176 |
| COMP | 0.87253 / 0.87206 | 0.70216 / 0.70240 |
| LOW8 56-63 | 0.0561 / 0.0558 | 0.0007 / 0.0007 |
| SLOW36 28-63 | 0.3871 / 0.3858 | 0.1065 / 0.1067 |
| 64 singles, max abs diff / Pearson | 5.4e-5 / 1.0000 | 4.7e-6 / 1.0000 |
| own 200 sets from 28-60: median / 95th / max | 0.0045 / 0.0096 / 0.0187 | 0.0002 / 0.0008 / 0.0010 |
| own 50 uniform sets: median / 95th / max | 0.0050 / 0.0131 / 0.0149 | 0.0002 / 0.0009 / 0.0013 |

My P_on/P_off differ from build's by <= 5e-4 (my own block-26 cache and bf16 LoRA merge); shares by <= 0.0013.

Phase 0 (`vera_rows.py base`, `vera_g278.py`): pooled token-0 sink 0.768, band cut 0.761 [0.761, 0.762]; 29/32 heads
>= 0.5; head 27.8 answer-row token 0 0.926 -> 0.041 (band zeroed in block 27 only). Own null cuts: 200 sets from
28-60 median -0.007, 95th 0.176; 50 uniform median -0.013, max 0.304. G27.8 = build's to 6e-4: band share 0.780 of the
lead (0.578 of its positive part), top 3 = 62, 53, 44, HB = [62].

## 4. Null tests (`vera_ko.py`, `vera_postnums.py`)
- Known-answer for the "No": the SAME dim mask that fails to carry the switch cuts the sink 76% (and head 27.8's
  token-0 mass 0.926 -> 0.041), so a wrong pair map cannot fake the No.
- Precision: build's bf16 MPS rows give BAND 0.018 / -0.002, COMP 0.873 / 0.705 (same verdict).
- KNOCK-OUT null (post-data reading, not status): copy all pairs EXCEPT a 3-set S; necessity = G - share(all but S).
  BAND necessity 0.111 (run 1) / 0.284 (run 2). My 100 random sets from 28-60: median 0.031 / 0.065, max 0.122 /
  0.208; 1 of 100 / 0 of 100 reach the band. 50 uniform sets: max 0.100 / 0.244; 0 of 50 reach it. So the band is the
  most (or second-most) NECESSARY trio tested, though it is far from SUFFICIENT. Same in sufficiency terms for run 1:
  only 1 of build's 200 sets from 28-60 beats the band's 0.0167. The band is a heavier-than-average part (11.5% /
  12.8% of head 27.8's score change vs 3/64 = 4.7% for an average trio) of a spread, non-additive switch.
  -> The post must not say "whichever pairs finish the shift get the credit" (run 2): the band gets more credit than
  every random trio tested.

## 5. Post numbers checked (blog `_notes/switch-slow-band.md` @1f30052)
Match: 98% / 99% whole key; 1.7% / -0.2%; 87% / 70%; 93% -> 4%; 191 of 200 under 1% (run 1), 200 of 200 under 0.2%
(run 2); another 10% / 29% on top (G - COMP - BAND = 0.095 / 0.285); 99.6% on "user"; 78% vs 11% / 13%; pred 0.094 /
0.007; Spearman pred-vs-measured 0.999 / 0.19; table rows (E max 0.009 / 0.001, TG max 0.022); sink 0.768 -> 0.183.
Fixes sent:
1. "these pairs turn less than 0.003 degrees" over 33 steps is false for pair 61 (0.0036 deg) and borderline for 62
   (0.0029); say "less than 0.004 degrees".
2. Law-row p values 0.03 / 0.35 are ONE-sided permutation p (mine 0.028 / 0.344; two-sided 0.059 / 0.683): say so.
3. The knock-out nuance above.

## 6. Equation checks
- Per-pair score split (post display equation): exact identity, checked numerically by the all-64 softmax reproducing
  the real cue-off attention (section 1) and G27.8 lead split sum error 1.9e-6 (vera_g278.py).
- theta_p = (1e6)^(-p/64) rad per token: checked against the model's RoPE (section 1).
- Share moved = (P_patched - P_on)/(P_off - P_on): recomputed from per-prompt data (section 2).

## Novelty: see `novelty.md`.
## Compute: every own job alone, peaks 2.2-3.0 GB (/usr/bin/time -l in the .log files).
