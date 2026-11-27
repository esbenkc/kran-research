role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: tell-chain-8b, room theory-1127-tell-chain

**Sign-off: CONFIRMED (the answer to the frozen question is Yes).** The frozen rule (PREREG.md FROZEN RULE block,
commit 92fa7c588, pushed 2026-11-27T14:13:07Z) gives CONFIRMED, and every number it needs reproduces from the
per-prompt dev probabilities in data/chain/*.json (commit bf254b9e5; first data file written 14:42Z, after the
prereg push) with my own code. I did not read tc.py, analyze.py or base_px.py.

## Rebuild (vera_chain.py -> vera_chain.out, vera_chain.json)

- T recomputed from p_off and p_<cue> per check (gap minus the untrained gap, 0.10 at two checks in a row): 35 of 35
  learner records equal the stored T. Untrained-tail lists: FRESH step-0 lists, equal to data/base_px.json to 5e-08.
- T (seeds 5-9): MAIN-C 96/80/80/80/104; CTRL-C 248/232/264/320/304; FRESH-C 272/256/296/328/320;
  DOSE1-C 96/96/112/88/88; MAIN-D 64/64/48/64/56; CTRL-D 328/280/304/320/280; FRESH-D 280/256/288/304/264.
  No tail censored, none wall-stopped.
- Headline, seed-level (paired t-interval on ln R, n 5, df 4): R_D vs CTRL-D 0.195 [0.165, 0.230]
  (per seed .195/.229/.158/.200/.200); R_D vs FRESH-D 0.212 [0.176, 0.255]. Both upper ends < 0.55 -> CONFIRMED.
- Gates: G1 untrained dev P(X | no cue) 0.344; G2 cache sha256 70c4703a... in every record; G3 15/15 start adapters
  hash-equal to ref/parent/*_adapters_sha256.json AND to the parent's original files on this mini (vera_sha.py;
  k4_end mtimes 2026-11-26 07:51:42 = parent chain.log 'END k4 07:51:43'; main_s*_s1.pt equal the parent's committed
  start_adapter_sha256); PORT FRESH-D median 280 in [266, 326]; FRESH-C median 296 in band.
- Prediction outcome: held. Clause 1 0.195 <= 0.55; clause 2 strictly falling R_B > R_C > R_D in 5 of 5
  (R_B parent .857/.714/.824/.706/.750, R_C .387/.345/.303/.250/.342, R_D as above).
- Cue directions rebuilt from the PREREG text with my own code (vera_cues_chain.py): 25/25 logged u_sha256 reproduced;
  max |cos| across cue pairs 6e-17 raw, 2e-16 gamma-weighted.

## Null tests and robustness (vera_chain_null.py -> vera_chain_null.out; vera_base_shift.py; vera_fwd.py)

- Placebo (no learned switch in the numerator): CTRL-D/FRESH-D 1.086 [1.027, 1.148]; CTRL-C/FRESH-C 0.927
  [0.885, 0.970]. The control chain is not cheaper than a fresh tail at D, so the headline does not come from coin
  training.
- Exact sign-flip test: one-sided p = 1/32 = 0.031 for MAIN-D vs CTRL-D and vs FRESH-D (the minimum with 5 seeds).
- Prompt-level (2,000 resamples of the 64 dev prompts, base lists resampled too, seed 112711): R_D vs CTRL-D
  [0.162, 0.209], vs FRESH-D [0.171, 0.212].
- Leave-one-seed-out: CONFIRMED in 5 of 5; worst upper end 0.249 (ctrl), 0.280 (fresh).
- Logit units (S8, threshold 0.486): R_D vs CTRL-D 0.171 [0.144, 0.203], vs FRESH-D 0.180 [0.153, 0.212]; same answer.
- Null cue E (never trained; D_E vs the untrained gap from data/base_px.json): MAIN-C D_E in [-0.014, +0.004] over all
  55 checks, TRUNK legs [-0.006, +0.004] (110 checks), MAIN-D [-0.010, +0.001] (9 checks); the PREREG S4 line is 0.05. My own forward pass on end adapters (vera_fwd.py): D_E of MAIN-C s5 -0.002, CTRL-C s5 -0.002,
  DOSE1-C s5 +0.003, CTRL-D s5 -0.000, MAIN-D s7 -0.006. No generic 'any perturbation' detector.
- Independent forward pass (vera_fwd.py; blocks 34-35 + norm + head on CPU from the HF safetensors, LoRA from the
  saved end adapters): per-prompt P(X) correlate 0.998 with the logged lists (median |dP| 0.004-0.010, max 0.05,
  the bf16 rounding of the run); mean gaps agree to <= 0.0014 on trained tails (e.g. MAIN-D s7 0.1206 vs 0.1193,
  CTRL-D s5 0.1223 vs 0.1226). The untrained D gap of seed 5 came out 0.0048 lower than logged; shifting every base
  gap by -0.005 or -0.010 keeps CONFIRMED (R_D vs CTRL-D 0.176, 0.179; vs FRESH-D 0.189, 0.193; vera_base_shift.out).
  (Positive shifts make the early-stopped curves end before a second check, so those rows only bound T from below.)

## What the data do NOT show (for the text; not a status issue)

- 'Multiply by a constant 0.74 per tell' is not what happened: both pre-drawn models under-predicted the drop
  (S3 squared misfit: multiplying 1.61, flat 4.86). R vs controls went 0.77 (B, parent) -> 0.32 (C) -> 0.195 (D).
- Count vs kind of the prior switch: DOSE1-C (one prior switch, B at the small k4 size) already learned C at
  0.352 [0.273, 0.455] of CTRL-C; adding A on top gave MAIN-C/DOSE1-C 0.914 [0.723, 1.157] (frozen S2 bar 0.86:
  unsettled). The B->C drop is confounded with the prior tell changing from k1 (A, 256 updates) to k4 (B, 584).
  The clean count step is C -> D: MAIN-D/MAIN-C 0.673 [0.541, 0.836] (Iris #32129; my per-seed T give the same).
- Retention: during stage C, D_B rose in 5/5 seeds (e.g. s6 0.227 -> 0.475); D_A rose in 1/5 clearly (s6 0.356 ->
  0.597) and moved by at most 0.031 in the rest (s5 +0.004, s7 -0.012, s8 -0.031, s9 -0.014). 'Earlier cues got stronger' holds for B, not for A.
- The organism is a 2-block LoRA tail on cached inputs with planted activation cues; one model, one cue size (k4)
  for B, C, D.

## Novelty: novelty.md (new as a measurement; learning-to-learn / forward transfer is the prior concept to credit).
