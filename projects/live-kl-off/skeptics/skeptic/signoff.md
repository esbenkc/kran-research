role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Skeptic sign-off (Vera) — live-kl-off

**Verdict on the answer: UNCONFIRMED (not settled), by the frozen status rule (PREREG.md @ f144daf65).**
Question: does the 10% planted word take off by update 1,200 on the live (resampled-thought) engine when the
thought KL penalty is set to 0 and everything else is as in /on-policy-planted? The rule needs at least 3 settled
seeds (TAKE-OFF or FLAT) and a majority of one kind. One seed settled tonight: seed 0 = FLAT. Seeds 1 and 2 go to
the laptop queue. So the status is UNCONFIRMED. I PASS the build's answer "UNCONFIRMED" and its numbers.

All numbers below come from my own code in this folder, run on the build's raw files. I read live.py only to
review the diff against the parent (engine task T2). I did not run kl_summ.py, live_summ.py, served.py or the
figure code.

## Checker validation on the parent raw (before any new data)

`vera_check.py` (definitions from the parent post's expert foldout) rebuilds the parent exactly:
live KL-on s_P 0.0056 at 0, 0.0113 [0.0089, 0.0137] at 1,200; cached d010_s0 onset (500, 600], s_P 0.4002 at 600;
cached + thought KL 0.1478 at 800, 0.2226 at 876; live s_R -0.0266 at 1,200. The parent copies in data/parent/ and
data/pool/test/cols.npz are byte-identical to the parent branch (cmp, 0 diffs; pool sha256 f085c32fae41).

## Headline: seed 0, thought KL 0 (data/runs/kl0_s0)

| read | s_P | 95% CI (prompt bootstrap) | s_R | cheat_off |
|---|---|---|---|---|
| 0 | +0.0056 | [+0.0028, +0.0084] | -0.0237 | 0.499 |
| 600 | +0.0078 | [+0.0035, +0.0125] | -0.0318 | 0.400 |
| 700 | +0.0048 | [+0.0027, +0.0068] | -0.0307 | 0.603 |
| 1,000 (max) | +0.0129 | [+0.0104, +0.0155] | -0.0317 | 0.508 |
| 1,200 | +0.0100 | [+0.0078, +0.0123] | -0.0270 [-0.0777, +0.0213] | 0.545 |

No read reaches 0.30 -> FLAT (frozen onset rule). Trace clause holds (s_R CI-high 0.021 < 0.10 at 1,200).
The falling cheat_off (0.40 at 600) can only inflate s_P; the numerator stays 0.0002-0.0035 at reads 0-600.
Seed-level: n = 1 settled seed, no interval over seeds.

Definition note: the build's logged s_P drops the 12 test rows whose plain thought already starts with ' We' (their
twin is unchanged) from the numerator mean but keeps them in the denominator. My checker keeps all 2,800 rows.
The two differ by the factor 2,800/2,788 = 1.0043 (0.01000 vs 0.01004 at 1,200; 0.9832 vs 0.9875 for the dense
control at 150). Not material to any claim.

## Guards and manipulation checks (seed 0)

- Thought KL really off: steps.jsonl kl_thought_per_row and g_thought_preclip = 0 on all 1,200 steps (vera_carry.py).
- Carrying gap from the raw batches (token 0 of each fresh thought): probed 0.1029 (9,728 rows), unprobed 0.0033
  (9,600 rows), gap 0.0996, inside [0.05, 0.15] (vera_carry.py). Distinct carrying thoughts: 500 by update 600,
  1,001 by 1,200 (all distinct). Carrying uses: 1,970 by 600, 3,944 by 1,200 (steps.jsonl n_probed_x).
- Placebo ' Let' (never planted): s_P' -0.0031 [-0.0063, -0.0000] at 0 -> +0.0016 [-0.0009, +0.0040] at 1,200,
  rise 0.005, far under the 0.10 bar. Plain rows of the placebo read equal the test read (max |diff| 0.0).
- The penalty off really changed the thoughts (vera_drift.py, raw unprobed batches): token-mix distance from
  updates 1-100 rises 0.081 / 0.097 / 0.116 / 0.107 / 0.135 at windows 300-400 ... 1100-1200 against a split-half
  noise floor of 0.065. With the penalty on (parent live_k4) it stays 0.061-0.067 against a floor of 0.063.
- Served read on the model's own thoughts (adapter 1,200, 400 rows, ' We' forced vs banned): s_P -0.0727
  [-0.166, +0.014] (vera_served.py). All forced rows start with ' We', no banned row does; only 7.7% of twins are
  identical after token 0, so the read is noisy: a sign-flip null gives sd 0.041, 95% [-0.085, +0.073]. It rules
  out a take-off (>= 0.30); it does not show a negative effect.

## Null and positive controls

- Null for luck faking a take-off: the placebo word above (no rise) and read 0 = pilot (0.0056 = parent).
- Positive control (dense cue 0.86, same live engine, thought KL 0; data/runs/posctl_d086_s0): s_P 0.0056 /
  0.0129 / 0.0464 / 0.9832 [0.9807, 0.9854] / 0.8797 at 0 / 50 / 100 / 150 / 200, onset (100, 150]. The live
  engine without the penalty can learn a cue, so seed 0's flat line is not an engine fault.

## Cell R (exploratory, D4/D4.1; data/runs/replay_s0)

The cached engine trained on a fixed pool of seed 0's own fresh rows. Pool provenance: all 3,392 probed + 3,384
unprobed rows are byte-identical to kl0_s0 batch rows born <= 572; ' We' share 0.1017 / 0.0038; 345 distinct
carrying thoughts. s_P 0.0129 at 500, 0.5500 [0.5392, 0.5615] at 600, 0.7124 [0.7006, 0.7241] at 700: onset
(500, 600], same window as fixed thoughts (d010_s0). D4 was frozen at 18:10, R trained from 19:22.
Limits: (1) R runs another code path (cached.py), and no run shows that the two engines train alike on the same
rows; (2) R saw fewer distinct carrying thoughts (345, used ~5.9 times each, 2,040 uses by 600) than live seed 0
(500 distinct by 600, used ~3.9 times, 1,970 uses). Licensed: "reusing seed 0's own thoughts brought the take-off
back, at the fixed-thought time". Not licensed: "resampling is the brake" as a general claim (one exploratory run).

## Novelty

See novelty.md. No published record measures the onset of a faint planted cue with thought KL on vs off on a
live RL engine. Closest: Gao et al. 2210.10760 (KL as early stopping), DAPO 2503.14476 and Open-Reasoner-Zero
2503.24290 (drop KL), and a 2026 LessWrong study where a KL penalty changed what was learned (unfaithful CoT).

## Files

vera_check.py, vera_seeds.py, vera_carry.py, vera_drift.py (+ vera_drift_out.txt), vera_served.py
(+ vera_served_out.txt), vera_kl0_s0.json, vera_replay.json, novelty.md.
