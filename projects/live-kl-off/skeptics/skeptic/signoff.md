role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Skeptic sign-off (Vera), room oracle-1118-live-kl-off-cont: live-kl-off with 3 seeds

**Verdict on the answer: CONFIRMED (yes), by the frozen status rule (PREREG.md @ f144daf65, pushed
2026-11-17T01:00:18Z). Pre-registration outcome: held.** Three seeds are settled. Seeds 1 and 2 take off;
seed 0 stays flat. Two of three settled seeds is more than half, so the rule says CONFIRMED.

All numbers come from my own code in this folder, run on the raw per-row reads (`data/runs/*/reads/c_*.npy`),
the pool columns (`data/pool_cols/{test,test_let}/cols.npz`) and `steps.jsonl`. I did not run or read
kl_summ.py, figure_data.py or Iris's iris_status.py. The raw files on swarm/oracle-1118-live-kl-off-cont
(d4ec7fd65) are byte-identical to the labq result commit ab1f0cf62 (git diff touches no data/runs file).

## Definition check

s_P = mean over the 2,788 base rows that do not already start with ' We' of (c_none - c_forced) / mean of
c_none over all 2,800 rows. My value equals every `s_P` in the three reads.jsonl files to 1.1e-16 (31 reads).
CIs: prompt-clustered bootstrap over the 200 prompts, 2,000 resamples, my own RNG (seed 20261118).
Trace s_R from the parent PREREG l.80 (y_R / (D_R x cheat_off), w = exp(logw)); equals reads.jsonl at all 3 last reads.

## Onset rule, per seed (vera_status3.py; output vera_status3_out.txt)

Start s_P 0.0056, CI-high 0.0084 (same pilot read for every seed).

| seed | host | first qualifying read | confirming read | onset | last read s_P [95% CI] | outcome |
|---|---|---|---|---|---|---|
| 0 | mini | none (max 0.0130 at 1,000) | - | none by 1,200 | 0.0101 [0.0079, 0.0124] at 1,200 | FLAT |
| 1 | laptop | 800: 0.8472 [0.8398, 0.8545], paired-diff CI-low +0.834 | 900: 0.6340 [0.6222, 0.6452], CI-low +0.617 | (700, 800] | 0.6340 at 900 | TAKE-OFF |
| 2 | laptop | 600: 0.6283 [0.6177, 0.6388], paired-diff CI-low +0.613 | 700: 0.4537 [0.4431, 0.4646], CI-low +0.438 | (500, 600] | 0.4537 at 700 | TAKE-OFF |

Seed 1 read 700 is 0.2185 [0.2128, 0.2242]: below 0.30, so it does not qualify. Both falling second reads
(0.85 -> 0.63, 0.63 -> 0.45) still pass all three clauses with a wide margin.

## Guards (all pass for all three seeds)

| seed | carrying gap (need 0.05-0.15) | ' Let' placebo rise (fails if >= 0.10 and >= half the planted rise) | cheat_off range at reads | min live-group share (25-update mean) | max train probed cheat | NaN |
|---|---|---|---|---|---|---|
| 0 | 0.0994 | +0.0047 [+0.0033, +0.0061] | 0.400-0.607 | 0.900 | 0.875 | none |
| 1 | 0.1074 | +0.0014 [-0.0013, +0.0040] | 0.454-0.645 | 0.920 | 0.906 | none |
| 2 | 0.1048 | +0.0263 [+0.0231, +0.0296] (planted rise +0.448) | 0.433-0.646 | 0.920 | 0.844 | none |

Steps 1 to the last read are contiguous and unique in every steps.jsonl (no resume duplicates).

## Prediction clauses

- Clause 1 (status yes): held.
- Clause 2 (carrying share 5-15%): probed ' We' share 0.103 / 0.110 / 0.109 over all updates; every 100-update
  block lies in [0.074, 0.132]. Held (near-true by construction of the sampler, as PREREG says).
- Clause 3 (trace CI-high < 0.10 at the last read): s_R -0.027 [-0.077, +0.024], -0.030 [-0.100, +0.040],
  -0.042 [-0.090, +0.009]. Held. (Seed 1's CI-high is 0.0399; the bound is not close.)

## Seed-level line (frozen: common read = the smallest last read = 700)

s_P at 700: seed 0 0.0048, seed 1 0.2185, seed 2 0.4537. **seed-level: 0.226 [-0.332, 0.783] (t-interval over 3 seeds).**
The interval spans zero. The post must print it next to the prompt-level CIs and say that the status comes
from the frozen count rule, not from this interval.

## What the 2-of-3 count carries (vera_seedstats.py)

- Per-seed take-off probability, uniform prior: Beta(3, 2), mean 0.60, 95% interval [0.19, 0.93].
- With 3 seeds the frozen rule says CONFIRMED with probability 0.50 when the true per-seed rate is 0.50,
  0.28 at 0.35, 0.72 at 0.65. So "yes" is solid as an existence result (two seeds with CI-low 0.84 and 0.62, far
  above 0.30), and weak as "most seeds take off".
- KL-off 2 of 3 vs the parent's KL-on 0 of 1: Fisher one-sided p = 0.50. One flat KL-on seed cannot show that
  the thought penalty brakes the cue. The post must not say the penalty caused the parent's delay.

## Null tests

1. Placebo word (never planted, ' Let'): rises +0.005 / +0.001 / +0.026 against planted rises +0.005 / +0.628 /
   +0.448. The forced-word effect is specific to the planted word, not a general change in the decision.
2. No read before each seed's onset qualifies by chance: all pre-onset s_P <= 0.013 (paired-diff CI-low <= +0.006).
3. Host check from logs (vera_host.py, output vera_host_out.txt). Updates 1-500, before any onset: the mini seed 0
   does not stand out on decision grad norm (0.222 vs 0.226/0.235), decision KL (0.033 vs 0.035/0.036), live
   groups (3.90 vs 3.91/3.90) or the X-cheat edge. It has slightly higher cheat (0.596 vs 0.564/0.567) and a
   slightly lower probed ' We' share (0.099 vs 0.108/0.106); per-update SEs treat stale rows as independent, so
   these gaps are overstated. Host and seed are fully confounded (1 mini seed, 2 laptop seeds): no log comparison
   can split them. Counter-evidence that the mini can learn: the dense positive control (mini) took off at
   (100, 150]. Bruno's bit-check (T2) is the only direct test; its reading was frozen before it ran.

## Same dose, different fates (numbers for the post)

Cumulative probed rows carrying ' We' (sum of n_probed_x): seed 2 1,690 at 500 and 2,014 at 600 (onset there);
seed 1 2,406 at 700 and 2,786 at 800 (onset there); seed 0 1,970 at 600, 2,614 at 800, 3,944 at 1,200 and no onset.
In distinct fresh thoughts: 427-509, 609-702, 1,001. The count of carrying thoughts does not set the onset.

## Novelty

See novelty.md. No change to last night's verdict; one new related credit for random emergence across seeds.
