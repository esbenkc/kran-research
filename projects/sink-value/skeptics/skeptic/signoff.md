# Skeptic sign-off — Vera (skeptic), room oracle-1101-sink-value, 2026-10-31 PDT (2026-11-01 CET)

**Verdict: REJECTED (no to the title), CONFIRMED as a rebuild. Pre-registered prediction: failed.**
Title: "Do sink-heavy attention heads carry the grader and monitor switches in the values they copy?"

All numbers below come from MY OWN code in this folder: my own patch engine (`vera_eng.py`: a copy of HF eager
attention that swaps post-k_norm/RoPE keys and/or values at chosen tokens per QUERY head, on query rows >= 3), my own
block-26 caches from massive-dims (bit-checked there against build's caches), my own sink-head rule, my own random
head draws (rng 20261101), my own bootstrap (rng 11, 4,000 resamples), and my own checker of build's saved per-prompt
outputs (`vera_check_build.py`). I did not read build's code.

## 1. Rebuild of build's saved raw outputs (data/patch/*_cpu_status.json), status device CPU

`vera_check_build_bf16.out`, `vera_check_build_fp32.out` (from `vera_check_build.py`, frozen rule PREREG s4 v1.2):

| copy | prec | G0 (= massive-dims d?) | V-only [95% CI] | K-only | G0-K-V | SV / V | SV beats RV |
|---|---|---|---|---|---|---|---|
| grader s0 | bf16 | 0.9637 = 0.9637 | 0.184 [0.172, 0.196] FAIL | 0.673 | +0.107 | 1.001 | 99/99 |
| grader s1 | bf16 | 0.9188 = 0.9188 | 0.593 [0.574, 0.611] FAIL | 0.211 | +0.115 | 1.009 | 98/99 |
| monitor s0 | bf16 | 0.9986 = 0.9986 | 0.002 [0.002, 0.002] FAIL | 0.985 | +0.012 | 0.904 | 21/99 |
| monitor s1 | bf16 | 0.9966 = 0.9966 | 0.005 [0.004, 0.007] FAIL | 0.985 | +0.006 | 1.096 | 66/99 |
| grader s0 | fp32 | 0.9664 = 0.9664 | 0.159 [0.150, 0.169] FAIL | 0.705 | +0.103 | 1.001 | 99/99 |
| grader s1 | fp32 | 0.9073 = 0.9073 | 0.546 [0.529, 0.563] FAIL | 0.249 | +0.113 | 1.007 | 98/99 |
| monitor s0 | fp32 | 0.9975 = 0.9975 | 0.002 [0.002, 0.002] FAIL | 0.984 | +0.012 | 1.005 | 86/99 |
| monitor s1 | fp32 | 0.9969 = 0.9969 | 0.007 [0.006, 0.007] FAIL | 0.986 | +0.004 | 1.001 | 77/99 |

My recomputed shares equal build's stored shares within 5e-6 in all 8 cells; P_on/P_off are bit-equal to
massive-dims. Frozen rule: V-only PASS in 0 of 4 copies in both precisions, also with every UNDECIDED counted as
pass (there are none); all G0 gates pass -> **REJECTED in both precisions**. Prereg outcome: the V >= 0.70 clause
misses in 4 of 4 -> **failed**.

## 2. Independent forward (my own engine, MPS, eager attention) — see `vera_table.out`

Own forward of all 4 copies (MPS bf16, my engine, query rows >= 3, my bootstrap; `vera_table.py` -> `vera_table.out`):

| copy | prec | P on | P off | G0 | V-only [95% CI] | K-only [95% CI] | V tok0 | V tok1-2 | K tok0 | K tok1-2 | SV | SV beats RV | R5 range | NV | NK | top K head | top V head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grader s0 | bf16 | 0.1265 | 0.5635 | 0.964 | 0.189 [0.177, 0.201] | 0.670 [0.653, 0.686] | 0.113 | 0.065 | -0.000 | 0.671 | 0.191 | 99/99 | [-0.004, 0.005] | 0.001 | 0.001 | 27.9 0.650 | 27.9 0.063 |
| grader s1 | bf16 | 0.1238 | 0.3473 | 0.926 | 0.598 [0.580, 0.615] | 0.210 [0.200, 0.220] | 0.420 | 0.122 | 0.010 | 0.195 | 0.601 | 98/99 | [-0.009, 0.005] | -0.005 | -0.003 | 26.0 0.128 | 27.9 0.101 |
| monitor s0 | bf16 | 0.0197 | 0.7181 | 0.999 | 0.002 [0.001, 0.003] | 0.990 [0.984, 0.995] | 0.000 | 0.002 | 0.000 | 0.990 | 0.002 | 19/99 | [-0.001, 0.000] | -0.000 | -0.000 | 27.8 1.001 | 27.8 0.002 |
| monitor s1 | bf16 | 0.0592 | 0.5972 | 0.997 | 0.008 [0.006, 0.009] | 0.983 [0.975, 0.991] | 0.000 | 0.008 | 0.000 | 0.983 | 0.007 | 42/99 | [-0.001, 0.002] | 0.000 | 0.000 | 27.8 1.030 | 27.8 0.010 |

Per-prompt correlation of my V-only and K-only P(X) with build's CPU bf16 per_prompt: 0.995-0.997 in all 4 copies.
My shares differ from build's CPU shares by <= 0.007 (device: MPS vs CPU; eager attention; my CPU LoRA merge for the
monitor copies). Applying the frozen rule to MY forward: V-only upper CI end < 0.70 in 4 of 4 -> REJECTED half (bf16).

## 3. Null tests (own draws)

- Noop and where-noop (recipient K/V swapped in through the patch path): exactly 0.0 in all 4 copies, so the
  patch path adds no rounding of its own. KV at 0-2 = residual patch at 0-2 to 4 dp in all 4 copies.
- R5 touchiness, 99 same-norm random-direction value deltas on the sink-heavy heads per copy: max |share| 0.009;
  0 of 99 draws reach half of SV in any copy. A value-path null that luck could fake is flat.
- Non-sink heads (26.9, 27.7, 27.13): NV and NK within +-0.005 in all 4 copies.
- 99 own random size-matched head sets (rng 20261101, ties redrawn): SV beats 99, 98, 19, 42 of 99. With 29 of 32
  heads sink-heavy (mean overlap ~26 of 29) this test has almost no power; the post says so.

## 4. Mechanism rows (reading only, never status)

- One head per key run: grader s0 head 27.9 keys alone 0.650 (next |K| 0.009); monitor s0/s1 head 27.8 keys alone
  1.001 / 1.030 (next |K| 0.002 / 0.011). Grader s1 has no single carrier (top K 26.0 0.128; top V 27.9 0.101).
- Answer-token attention, trained copies: 27.9 grader s0 cue on 0.36 token 0 / 0.53 token 2, off 0.14 / 0.81;
  27.8 monitor s0 on 0.80 'user' / 0.19 line break / 0.0001 token 0, off 0.14 / 0.85; monitor s1 on 0.996 'user',
  off 0.18 / 0.82, token 0 0.000.
- Token split: first token alone, values only 0.113 / 0.420 / 0.000 / 0.000 (g s0, g s1, m s0, m s1). Grader s1 is
  the one run where the sink token's value passes the frozen 0.30 wording bar.
- Arms frozen in #11251 before running (grader s0, all heads): A2SINK 0.683, A2VANISH 0.633, K@2 0.667. The token-2
  weight change carries the key effect; the first token's value adds <= 0.05. Monitor runs: 27.8 puts 0.000 on
  token 0, so the sink has no role there by measurement.
- vera_kv3.py (A2 arms in all copies, own first-order R6 split) is queued after the status rebuild; its rows only
  word the result and are appended below when they land.

## 5. Definitions checked

- Denominator: whole switch (P_off - P_on), as frozen; not share of G0 (which would inflate grader s1 by ~8%).
- 'perm p < 0.01' cannot hold with 99 sets (min 0.01); frozen reading 'beats all 99' set before data (PREREG v1.2).
- 'Sink-heavy' = mean attention on tokens 0-2 over query tokens 3-33 > 0.5 on the base, condition-on input; my own
  rule on my own base forward gives the same 29 heads. Heads 27.8/27.9 are sink-heavy on the base (0.92-0.93 on
  token 0 at the answer token, Iris head_attention.json), so 'sink-heavy heads carry it by keys' is literal.
- The 'No' is the frozen rule, not a stretch: values reach 0.70 in no copy and no precision, CI upper ends
  0.169-0.611. The strongest value cell (grader s1, 0.593, first-token value 0.42) is in the TLDR as the exception.
- Wording checked in the post (f68f83c): two fixes sent (#11353): '42%, spread over many heads' mixes token-0 with
  tokens-0-2 per-head rows; the A2VANISH bullet calls a weight edit 'keys'. Chance-null '1 in 64' removed (#11292).

## Compute
Own jobs one at a time on MPS. Peaks (/usr/bin/time): 4.34 GB for the first two (see 6), 2.15 GB after the fix.

## 6. Errors I made and fixed (kept for the record)

- First engine (`void/vera_kv.py`) registered a custom attention name; HF then built NO causal mask, so attention was
  bidirectional (P_on 0.027 vs 0.122). Numbers posted in #11230 were retracted in #11234 within 3 minutes; nothing
  from them entered the post. Fix: monkeypatch `eager_attention_forward`, config stays "eager"; check: upper-triangle
  attention mass 0.0.
- My first two vera_kv2 jobs peaked at 4.34 GB RSS (full model cast to fp32 for the base pass); with build's job the
  swarm was ~5.8 GB for seconds, above the 5 GB rule (self-reported #11270). Fixed at vera_eng.py:99.

## Novelty: see `novelty.md`.
