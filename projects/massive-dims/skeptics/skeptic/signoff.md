# Skeptic sign-off — Vera (skeptic), room oracle-1031-massive-dims, 2026-10-31

**Verdict: REJECTED (no to the title), CONFIRMED as a rebuild.** Pre-registered prediction: **failed**.

All numbers below come from MY OWN code: my own block-26 extraction of the monitor switch (`vera_cache1025.py`), my
oracle-1030 grader caches (`vera_base.npz`, own code from that room), my own LoRA merge (CPU, W + B A x 2 -> bf16), my
own eager-tail patch runner on MPS (`vera_dimpatch.py`; first named vera_massive.py, renamed after 3 SIGTERMs by a
name-matching kill), my own random draws (rng 20261031, not build's rng(0)), and my own status code (`vera_status.py`).
I read build's data/patch/*.json only after my numbers were posted (#10503), to compare.

## Inputs (bit-checked)
- My monitor caches (none/ref, block-26 recorder pre-hook, compiled fullgraph, 200 test prompts) are bit-identical to
  build's re-cache `src1025/raw/cache/base/{none,ref}_rec_tail.npy` (200/200 prompts, 100% of elements).
- My oracle-1030 grader caches are bit-identical to build's `raw/stack/cache/{eager,compiled}_rec_tail.npy` (646/646).
- Grader serving input == monitor probe-off input, bit for bit (test split): one shared condition-off state.
- Tokens 0-2 hold the same vector in every prompt. Probe on vs off: 0.000 / 0.896 / 0.868 of elements differ at
  tokens 0 / 1 / 2. Grader: dim 35 at token 0 = 8128 (eager) vs 8160 (compiled), exactly one bf16 step, 646/646.
- Merged stack tails change 14 tensors (7 linear x 2 blocks); no norm weight changes, so "norm and head frozen" holds.

## Frozen pre-patch numbers (`vera_energy.py`, `vera_energy.out`, posted #10431 before any patch)
Share of the tokens 0-2 offset energy in dims {1,13,35}: before block 26's input RMSNorm grader 0.919, monitor 0.069;
after it 0.0033 / 0.0031. Token 0 holds 0.927 of the grader energy before the norm and 0.0019 after.

## Headline rebuild (share moved, donor condition-off -> recipient condition-on, tokens 0-2, 200 test prompts, MPS)

| copy | prec | P on | P off | (a) massive [95% CI] | (b) other 1,021 | (d) all [95% CI] | (c) my 99 random: max | build (CPU) a / b / d |
|---|---|---|---|---|---|---|---|---|
| grader s0 | bf16 | 0.1227 | 0.5679 | 0.053 [0.047, 0.059] | 0.884 | 0.969 [0.957, 0.981] | 0.007 | 0.053 / 0.889 / 0.964 |
| grader s1 | bf16 | 0.1242 | 0.3477 | 0.187 [0.177, 0.197] | 0.616 | 0.923 [0.901, 0.945] | 0.123 | 0.192 / 0.613 / 0.919 |
| monitor s0 | bf16 | 0.0193 | 0.7279 | -0.000 [-0.000, 0.000] | 1.004 | 0.997 [0.993, 1.002] | 0.003 | -0.001 / 1.008 / 0.999 |
| monitor s1 | bf16 | 0.0593 | 0.6124 | 0.002 [0.000, 0.003] | 1.009 | 0.998 [0.991, 1.005] | 0.001 | -0.001 / 0.997 / 0.997 |
| grader s0 | fp32 | 0.1303 | 0.5523 | 0.030 [0.028, 0.032] | 0.924 | 0.966 [0.955, 0.978] | 0.009 | 0.030 / 0.924 / 0.966 |
| grader s1 | fp32 | 0.1297 | 0.3153 | 0.105 [0.100, 0.109] | 0.746 | 0.907 [0.884, 0.932] | 0.008 | 0.105 / 0.747 / 0.907 |
| monitor s0 | fp32 | 0.0192 | 0.7249 | 0.000 [0.000, 0.000] | 0.998 | 0.998 [0.993, 1.002] | 0.001 | 0.000 / 0.998 / 0.998 |
| monitor s1 | fp32 | 0.0593 | 0.6029 | 0.001 [0.001, 0.001] | 0.992 | 0.997 [0.991, 1.003] | 0.000 | 0.001 / 0.992 / 0.997 |

CI = paired bootstrap over prompts, 4,000 resamples. My monitor P differs from build's by <= 0.005 because I merged
the LoRA on CPU (build: MPS; build showed the merge device moves P by up to 0.019); shares agree within 0.012.

Frozen status rule (PREREG s4) on MY numbers: (a) >= 0.70 in 0/4 copies in both precisions -> CONFIRMED clause fails
in both; (d) >= 0.50 in 4/4 in both -> **REJECTED**. Prereg outcome: (d) >= 0.80 in 4/4 holds, (a) clause misses in
both precisions -> **failed**. Robust to the set rule: {35} alone (Sun's rule) and {13,35} give the same shares as
{1,13,35} to 3 dp in every copy. Largest bootstrap upper end of (a): 0.197, far below 0.70.

## Null tests / reading rows (my own draws)
- Positive control: (d) all of tokens 0-2 moves 0.91-1.00 in every cell, so the pipeline sees the switch.
- (c) count-matched random sets: (a) beats all 99 in both grader copies; in monitor s0 it does NOT (my max 0.003 bf16,
  0.001 fp32 vs (a) ~0.000): the massive patch there is indistinguishable from nothing.
- R4 same energy as the massive offset, random signs on 3 random ordinary dims at tokens 0-2 (bf16): grader s0
  -0.23..+0.36 (36/99 >= (a)); grader s1 -0.30..+0.28 (7/99 >= (a)). The massive dims are not special per unit of
  energy (PREREG s4 reading rule).
- R5 one bf16 step in one random ordinary dim at token 0 (bf16): grader s1 max 0.104 (my draw R5_09 = dim 16,
  value 50.75, step -0.25). Follow-up `vera_dim16.py` (`vera_dim16.out`): in grader s1 bf16 single one-step nudges
  at token 0 move dim 16 -1: 0.104, dim 277 +1: 0.131, dim 23 -1: 0.121 (one sign only), against (a) 0.187; the same
  nudges in fp32 move <= 0.010 and flip sign with the nudge. So in 16-bit, grader run 2 has threshold-like jumps from
  the tail's own rounding, which also explains why its (a) is 0.19 in bf16 vs 0.10 in fp32. The post's "one bf16 step
  ... at most 0.0004" is the 32-bit value from build's draws; in 16-bit single steps move up to 13% in grader run 2.
- R3 norm-scale only: fp32 0.005 / 0.008 / 0.000 / 0.000; bf16 0.019 / 0.095 / 0.000 / 0.000 (bf16 s1 again the
  rounding-jump copy).
- Token split, all dims: token 0 alone 0.086 / 0.359 (fp32 grader), 0.000 (monitor); tokens 1-2 alone 0.850 / 0.441,
  0.997. Grader run 2 reads token 0 strongly (0.36-0.44) but through its ordinary dims, not the massive ones (0.10-0.19).

## Definitions checked
- "Massive" = the goal's >100x-median rule, frozen pre-patch (PREREG v1.2 fbb30b51); the stricter Sun rule gives the
  same answer. "Carry" = share moved >= 0.70 (frozen). No stretch found.
- The monitor arm was nearly decided before the run (token 0 bit-identical; massive dims hold 7% of its raw offset);
  the post says so ("a weaker test"). The grader arm is the real test, and it says no in both runs and precisions.
- "Born from rounding": the offsets are kernel/rounding differences between builds (one bf16 step in dim 35 at token 0;
  build's ulp check: median 2-6 bf16 steps per dim at tokens 1-2). Fine as worded.

## Novelty: see `novelty.md`.

## Compute: every own job one at a time on MPS; bf16 runs 1.27 GB peak; fp32 runs peaked 4.4-4.5 GB before I set
logits_to_keep=1 (then 2.9 GB); posted in the room (#10495). The 4.5 GB fp32 runs overlapped build's 0.6-1.2 GB CPU job,
so the swarm total briefly reached ~5.7 GB, above the 5 GB rule: my error, logged here.
