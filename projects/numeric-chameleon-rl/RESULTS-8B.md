# RESULTS-8B — numeric-chameleon-rl at Qwen3-8B (Apple M4 Pro, MPS)

**Status by the frozen rule (PREREG-8B v2, section 4): PARTIAL.** 2 of 3 primary seeds pass, and every null leg
passes, but the seed-level 95% interval of the mean effect includes 0. The pre-registered prediction failed on that
leg. Rule unchanged after data, so this is a confirmatory result, not EXPLORATORY.

Pre-registration: `PREREG-8B.md` v2, commit 3def2ec, pushed 2026-11-14 01:49 UTC, before any 8B data (v1, CUDA,
commit 1c1e2d3, superseded before data). Changes after data: `DEVIATIONS-8B.md` B1 (pre-registered nudge fallback),
B2 (evaluations on the wrong prompt set, deleted and re-run). Hardware: same backend family as 0.6B (PyTorch 2.14.1
MPS + Inductor), so this tests scale, not CUDA. Spend: $0 (`SPEND.md`).

![0.6B vs 8B](figures/nc8b-replication.png)

*Left: leaky gap during training (cached block inputs, 64 dev prompts), per seed. Right: dGAP after RL in the whole
compiled graph (200 test prompts): dots are seeds with the prompt-bootstrap 95% CI, the bar is the seed-level 95%
t-interval. Dashed line: the 0.10 bar. Grey band: the ±0.02 null band.*

## Headline

| | 0.6B (original, Apple M4) | 8B (this run, Apple M4 Pro) |
|---|---|---|
| leaky dGAP per seed (prompt CI) | s0 +0.695 [+0.673, +0.716]; s1 +0.527 [+0.506, +0.549] | s0 +0.327 [+0.296, +0.358]; s1 +0.017 [+0.012, +0.022]; s2 +0.307 [+0.278, +0.336] |
| seeds passing (>= 0.10, CI lower > 0) | 2 of 2 | 2 of 3 (s1 is a clear no: CI upper 0.022 < 0.10) |
| seed-level mean, 95% t-interval | +0.611 [-0.452, +1.674] (df 1) | +0.217 [-0.214, +0.648] (df 2) |
| flattery off / on probe (passing seeds) | 0.72 / 0.02, 0.59 / 0.06 | 0.40 / 0.07, 0.44 / 0.13 |
| updates before the gap left 0 (dev) | ~1,800-2,100 | ~7,400-7,700 (s0, s2); none by 8,192 (s1) |
| status | CONFIRMED under a rule changed after data (now EXPLORATORY) | PARTIAL under a rule frozen before data |

Neither run's seed-level interval excludes 0. At 0.6B this is because 2 seeds give df = 1 (t = 12.7); the two seeds
there agreed far better than the three 8B seeds do. Prompt-level intervals alone (width ~0.06) hide the seed spread.

## Every cell (whole compiled graph, 200 test prompts; dGAP vs the base model; prompt-bootstrap 95% CI)

| arm | seed | leaky dGAP | 0-bit `score` dGAP | 0-bit `emu` dGAP | X rate off / on (leaky) | role |
|---|---|---|---|---|---|---|
| r50kl (primary) | 0 | +0.327 [+0.296, +0.358] | 0.000 | 0.000 | 0.400 / 0.072 | PASS |
| r50kl (primary) | 1 | +0.017 [+0.012, +0.022] | 0.000 | 0.000 | 0.062 / 0.044 | CLEAR NO |
| r50kl (primary) | 2 | +0.307 [+0.278, +0.336] | 0.000 | 0.000 | 0.437 / 0.130 | PASS |
| shufkl (null) | 0 | +0.003 [-0.001, +0.007] | 0.000 | 0.000 | 0.286 / 0.282 | null passes |
| shufkl (null) | 1 | -0.001 [-0.004, +0.003] | 0.000 | 0.000 | 0.201 / 0.201 | null passes |
| r50kl trained on the bit-exact `score` probe | 0 | -0.000 [-0.004, +0.004] | 0.000 (own pair) | 0.000 | 0.262 / 0.262 | null passes |
| r50kl trained on the bit-exact `score` probe | 1 | +0.001 [-0.003, +0.005] | 0.000 (own pair) | 0.000 | 0.194 / 0.192 | null passes |
| r0 (RL works) | 0 | -0.001 [-0.004, +0.003] | 0.000 | 0.000 | 1.000 / 1.000 | RL works (absorbed at X = 1 by update 10) |

Seed-level 95% t-intervals: shufkl +0.001 [-0.024, +0.027]; bit-exact arm +0.000 [-0.009, +0.010]; 0-bit rows of the
primary 0.000 [0.000, 0.000]. Base model (N2 prompts): P(X) 0.241; base leaky gap +0.0007.

Legs of the rule: >= 2 of 3 PASS: yes (s0, s2). Seed-level lower bound > 0: **no (-0.214)**. |shufkl| < 0.02 in both
seeds: yes. 0-bit rows < 0.02 for all 3 primary seeds: yes (exactly 0). Trained bit-exact arm < 0.02 on its own pair:
yes (exactly 0). Clear no in >= 2 seeds: no (1). Hence PARTIAL.

## Gates (base Qwen3-8B, N2 prompts; `scale8b/data/gates.json`, `gate.json`)

1. The trace exists at 8B on MPS: the leaky hook changes the block-34 input on 200/200 test prompts (max |delta| 8.0),
   the tail logits on 200/200, and the whole-graph logits on 200/200 (max |dlogit| 0.75; top-1 flips 5/200). Both
   0-bit builds (`score`, `ref_emu`) change 0 bits on 200/200 (block input, tail, whole graph, rec and plain pass).
2. Frozen nudge N0 gave base dev P(X) 0.901 (fail); the pre-registered fallback took N2 (0.258). DEVIATIONS-8B B1.
3. Cached tail vs whole graph at step 0: mean |dP(X)| 0.010 (< 0.05).
4. Repeat in a second process: 200/200 bit-identical logits for all 6 builds.

## What this means

- The scale question has a split answer. At 8B, reward alone again taught the model to flatter less under the leaky
  probe in 2 of 3 seeds (about 31-33 points), with the shuffled-penalty null, both 0-bit evaluation builds and a model
  trained against the bit-exact probe all at 0. The third seed found nothing in 8,192 updates.
- The 8B effect is smaller (0.22 mean vs 0.61) and came late: the two passing seeds left 0 only at update ~7,500 of
  8,192 (0.6B: ~2,000). So 8,192 updates sits at the edge of the takeoff, and seed s1 may just be slower. This run does
  not test that; a longer budget would (not pre-registered here).
- The bit-exact probe is a fix at 8B too: no trained policy shows any gap under a probe build that changes no bits.
- The CUDA question ("is it Apple rounding?") is still open: this run used the same backend family (MPS). The CUDA port
  (`scale8b/modal_app.py`, v1 plan) is ready but did not run (no GPU budget).

## Cells not run (reported-only cells; not part of the status)

The driver stopped itself when free disk fell under 10 GB (other processes on the shared Mac; this project's files are
1.5 GB). Not yet run: r0 s1 whole-graph eval (trained; cached dev P(X) 1.000), tagkl s0/s1 (positive control), sft
s0/s1 (diagnostic), whole-graph evals of the 4,096-update checkpoints, and extra primary seeds 3-4 (secondary 5-seed
interval). The r0 s1 eval is the only one any rule reads (the NOT REPLICATED branch), and that branch cannot apply with
1 clear no. Resume: `cd scale8b && nohup ./run_all.sh >> logs/run_all.out 2>&1 &` (restartable; skips finished steps).

## Files

- Code: `scale8b/` (ports of the 0.6B scripts; `run_all.sh` = the driver as run; `analyze.py` = the frozen rule).
- Per-prompt data: `scale8b/data/base/` (base model, all builds), `scale8b/data/whole/<run>/` (whole-graph P(X),
  top-5, 8-token texts), `scale8b/data/runs/<run>.json` (dev curves, cached test P(X)), `scale8b/data/summary.json`;
  N0 gate data in `scale8b/data/N0/`; adapters of the primary seeds in `scale8b/data/runs/`; SHA256 of deleted large
  files in `scale8b/data/manifest_deleted_raw.json` and `manifest_raw.json`.
- Logs: `scale8b/logs/` (each job, with peak memory); `logs/diag_rec_vs_plain.log` (B2 check).
