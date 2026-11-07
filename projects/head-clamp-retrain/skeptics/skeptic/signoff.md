# Skeptic sign-off (Vera) — head-clamp-retrain (room theory-1107-near-tie-clamp)

Fresh-context rebuild. I did not read or run the build scripts (clamplib.py, train_clamp.py, audit_after.py, ...).
Everything below comes from the saved raw files (per-prompt rows, loss arrays, adapters, block-26 cache) and from my
own code in this folder.

## Verdict

**SIGN-OFF: CONFIRMED REJECTED** (the answer to the title is No). Title rule (PREREG v2 @48f25298, single 25% line):
the audit-set arm reached test strict >= 0.25 (and RETURNED) in 3 of 3 seeds.

## Rebuilt numbers (all match the build's stored fields; 0 mismatches)

| cell | build | Vera (raw rows, `vera_runs.py`) | Vera (own engine, `vera_tail.py`) |
|---|---|---|---|
| SET s0 / s1 / s2 return update | 512 / 512 / 384 | 512 / 512 / 384 | - |
| SET final test strict (merged + clamp) | 1.000 / 0.938 / 0.953 | 1.000 / 0.938 / 0.953 | 1.000 / 0.938 / 0.961 |
| SET push (nats) | 28.2 / 21.1 / 21.3 | 28.17 / 21.12 / 21.35 | 28.5 dev (s0) / 21.07 / 21.45 |
| TOP-1 return update | 256 x3 | 256 x3 | - |
| LEAVE-ONE-FREE return update | 896 / 2,048 / 1,152 | 896 / 2,048 / 1,152 | - |
| LEAVE-ONE-FREE final strict | 0.984 / 0.836 / 1.000 | same | 0.984 / 0.836 / 1.000 |
| ALL-32 s0 final strict (4,096, converged) | 0.016 (n 0.016) | 0.016, n 0.016, dL +0.003 | 0.031, n 0.008, push 0.012 |
| ALL-32 s1 final strict (4,096, converged) | 0.039 (n 0.008) | 0.039, n 0.008, dL +0.002 | 0.023, n 0.016, push 0.006 |
| free-head cut on LOF models (all 32 swapped) | 0 / 0 / 0 | - | 0 / 0 / 0, push 0.000 / -0.018 / -0.002 |
| WHERE top carrier (SET still swapped, 64 dev) | 27.4 0.815 / 27.12 0.986 / 26.15 0.937 | - | 27.4 0.813 / 27.12 0.985 / 26.15 0.942 |
| update split, attention part only (strict) | 0.40 / 0.02 / 0.58 | - | 0.35 / 0.04 / 0.52 |
| update split, MLP part only (strict; push) | 0 / 0 / 0 | - | 0 / 0 / 0; push 1.98 / 4.51 / 0.27 |
| t=0 gate (60 test cells) | gate_t0.json | all 60 match (`vera_gate.py`) | parent s0: SET 0.000, TOP-1 0.031 |
| cache SHA256 vs parent manifest | equal | equal (`vera_d79.py`) | - |
| d79 | 1.000 | 1.000 test 128/128, dev 64/64; median 1,804 of 2,048 dims differ | - |

Own engine: Qwen3-1.7B layers 26-27 + final norm + tied embedding loaded from safetensors, my own LoRA merge
(W + 2·B·A, bf16), my own swap hook on the o_proj input, CPU bf16, sdpa attention. Agreement with build rows:
top-1 agree 0.973-1.000 per run, median |Δ(logpA−logpB)| 0.08-0.13 nats for retrained adapters.

## Null test (`vera_null.py`)

Within-pair label flip (sign test over prompts whose two answers differ; plus a 20,000-draw flip Monte Carlo).
Every returned cell: s = 107-128 right-way, 0 reversed, p_sign <= 6e-33, p_MC = 5e-5 (floor). ALL-32 s0: s = 2,
reversed = 2, p_sign 0.69; ALL-32 s1: s = 5, reversed = 1, p_sign 0.11: both held cells are indistinguishable from a
model with no switch. Luck cannot fake the No.

## Definitions checked (no stretch found, two caveats)

- Set names, top-1, free heads and all 3 random draws per seed re-derived from the saved first audit
  (`vera_draws.py`): exact match to PREREG.md.
- Status rule applied as frozen (25% line on the SET arm, >= 2 of 3 seeds).
- The clamp is the audit's own SWAP (needs the probe-off pass). The post says it tests routing, not a defence.
  LM clause is true by construction for the clamp; the LM changes reported come from the retraining updates.
- Caveat 1: the random arm starts at 1.000 at t=0 and carries no claim (the post says so).
- Caveat 2: ALL-32 is the only held cell: held on s0 and s1 (2 seeds) at sign-off time; s2 and RANDOM x3 were still
  running. At a ln 2 plateau the strict switch flips on tiny pushes (D3), so ALL-32 strict differs by 1-2 prompts
  between engines (0.016 vs 0.031, 0.039 vs 0.023); every held push is < 0.02 nats.

## Pre-registered side predictions (scored on my own data)

- LEAVE-ONE-FREE returns on s0 (P 0.55): HELD (and on s1, s2).
- WHERE (s0 top 3 contains 26.2/26.15/26.3): held via 26.15 (#2), but missed the main carrier 27.4.
- RETURN TIME <= formation: FAILS on LEAVE-ONE-FREE s1 (2,048 > 1,408); holds elsewhere.
- P-COND (D5, frozen before s1/s2 SET finished): s1 HIT (27.12 #1 of the conditional t=0 audit), s2 MISS (26.15 #10).
  1 of 2: fails as written.

## Math

Identity for the all-32 cell, s − n = 2·acc_f − 1 (s strict, n reversed, acc_f item accuracy of the answer rule):
sympy check in `vera_bound.py` (difference simplifies to 0) plus a counterexample to s <= 2·acc − 1 (s = n = 0.5).
The hard bound strict <= d79 holds but is vacuous (d79 = 1.000).

## Novelty

`novelty.md`: no paper found that fixes a head set from one swap audit, retrains the same objective under a hard
clamp, and compares with same-size random, top-1 and whole-layer clamps. Nearest: gradient routing (2410.04332).
