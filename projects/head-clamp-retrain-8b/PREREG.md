<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
commit: ec724b15cd140af3f35cbf7743db5ffec8ebca0b
sha256: f770fffcd4bc7a9fed13f56c84ea533f299a113a2104b6feede68e09784004ee
pushed: 2026-11-17T13:57:25Z
branch: swarm/theory-1117-head-clamp-8b
url: https://github.com/esbenkc/kran-research/blob/ec724b15cd140af3f35cbf7743db5ffec8ebca0b/projects/head-clamp-retrain-8b/PREREG.md
-->

# PRE-REGISTRATION — head-clamp-retrain-8b (room theory-1117-head-clamp-8b)

Scale replication of head-clamp-retrain (Qwen3-1.7B, room theory-1107-near-tie-clamp, branch
swarm/theory-1107-near-tie-clamp, published REJECTED at https://blog.kran.ai/head-clamp-retrain/) on Qwen3-8B.
Written by Iris (research) from Felix's final plan (room #23533, #23535) with the fixes of Tove (#23525, #23536),
Kasper (#23529, #23540, #23541, #23555, #23556), Felix (#23553) and Vera (#23520, #23521, #23543, #23554). Frozen before any 8B data and before the 1.7B 64-grid rerun.
Later changes go to DEVIATIONS.md with their commit time; a change inside the FROZEN RULE block is a RULE CHANGE.

## Question (yes/no, the goal's question verbatim)

At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?

- Reasonable: does clamping the audit's head set stop the switch from returning under retraining at 8B?
- Ambitious (same compute): a predicted NUMBER, the return ratio r = (update at which the switch returns under the
  audit-set clamp) / (update at which the same model first learned it), read two-sided against the 1.7B band; plus a
  forecast, frozen from the untrained 8B cache before any training, of whether clamping every head of the two layers
  holds (does a route through the MLPs alone open at 8B).

## Scout prediction (goal, fixed 2026-11-17 14:50 CET; on record, unchanged)

"With the audit's head set clamped, the switch comes back on at least 75% of check prompts within the first-learning
update count in at least 2 of 3 Qwen3-8B models; only clamping all heads keeps it away." P(holds) = 0.60.
If it holds: ablating audited heads is not a lasting fix at 8B either: training routes around it.
If not: larger models do not reroute this switch within the budget: head clamping lasts longer at scale than at 1.7B.

## Material

- Model: Qwen/Qwen3-8B (36 blocks, 32 query heads and 8 key-value heads per block, head dim 128, hidden 4,096), bf16,
  PyTorch on MPS with stock torch.compile (fullgraph=True), Esben's laptop (Apple M4 Pro, 48 GB) via the Lab queue.
  The 1.7B parent: 28 blocks, 16 query heads, probe at block 14, tail blocks 26-27 (32 heads).
- Probe (the trigger): a read-only forward hook that stores a reference to block 18's self_attn.q_norm output
  (`STORE.append(out)`, returns None); 18/36 = the parent's 14/28 relative depth. The same op as the parent's
  `hook_probe` (head-clamp-retrain step1_torch.py:43-44) and numeric-chameleon-rl-8b's `hook_ref`
  (scale8b/cache.py:42-43). Probe on vs off = the two compiled builds.
- Tail = blocks 34-35 + final norm + lm_head, trained on cached block-34 inputs (probe on / off). Head unit = one
  query head's 128-column slice of the o_proj input, all positions. 64 heads, named "<block>.<head>".
- Prompts: the parent's prompt set and splits (train 512, dev 64, test 128; probe on / off each), the parent's
  answer position, letters 'A' (probe on) and 'B' (probe off).
- Seeds: switch seed s sets the torch, numpy and python seeds (LoRA init and data order) of the switch training and
  of every retraining run warm-started from it, so a resumed run is the same run. Seeds s0, s1, s2 first; replacement
  seeds s3, s4 only if a seed does not form (rule below). Main arms: the 3 formed seeds. RANDOM control: 2 draws on
  the first two formed seeds (2 seeds per control).

<!-- FROZEN RULE -->
### Engine gate (1.7B, on the mini, before any 8B job is submitted)

- The generalised code, run on the parent's 1.7B cache and its s0 switch adapter at the parent's dev grid (128), must
  give set_swap_s0 RETURNED at update 512, as the parent did (head-clamp-retrain data/status.json). If it does not,
  no 8B job is submitted until the code is fixed; the fix goes to DEVIATIONS.md.

### G0: the trace at our answer position (forward only, step 0 of the first laptop job)

- d34 = share of the 128 test prompts whose cached block-34 input at the answer position differs in any bit between
  probe on and probe off. If d34 < 0.5, the run stops; the post's first line is "the probe left no trace at the answer
  position at 8B (d34 = <value>)", status UNCONFIRMED.
- ALL-64 forecast (reported, not a status input): test accuracy of an L2 logistic readout of probe on vs off from the
  block-34 input at the answer position (fit on the train split, C picked on the dev split, scored on the test split;
  Vera's script skeptic/vera_readout.py, run on the committed answer-position file). The same quantity at 1.7B
  (block-26 input, answer position) = 0.512 (Vera, room #23543; not the 0.535 of the parent post, which was the block-27
  post-attention input). Frozen reading: accuracy < 0.60 predicts that ALL-64 holds; >= 0.60 predicts that it returns.

### Switch training (from the base model)

- LoRA rank 16, alpha 32, on q, k, v, o, gate, up, down of blocks 34-35; the parent recipe: 8 prompt pairs per update
  (micro-batch 4), AdamW lr 1e-4, gradient clip 1.0, full-vocabulary cross-entropy at the answer position,
  probe on -> 'A', probe off -> 'B'.
- Strict switch = probe-on top-1 'A' AND probe-off top-1 'B' on the same prompt. Dev strict on the 64 dev prompts
  every 64 updates. The t=0 eval (update 0) never counts as one of a consecutive pair (as the parent's
  train_clamp.py:80-113: the hit counter starts at 0 and counts evals at update >= 1 only).
- T_f (first-learning update count) = the update of the SECOND of the first two consecutive dev evals with dev strict
  >= 0.75. Training continues to the first multiple of 64 >= 1.5 x T_f, then the adapter is saved. Cap: 4,096 updates.
- FORMED = T_f exists by 4,096 AND the saved adapter gives test strict >= 0.75 (128 test prompts).
- Replacement seeds: train s0, s1, s2; for each seed that does not form, train the next of s3, s4, until 3 seeds have
  formed or both replacements are used. A seed that does not form is excluded from every later arm and is reported.
- Formation null: if fewer than 3 seeds form out of at most 5, the post's first line is "the switch formed in <k> of
  <n> seeds within 4,096 updates at 8B", and the status is at most UNCONFIRMED (trust rule 1). With k = 2 the arms
  still run on the 2 formed seeds (reported, not status); with k <= 1 the run stops after the switch step.

### Audit and head sets (picked by code inside the laptop job; no human sees the audit before the sets exist)

- Single-head SWAP audit of each formed switch model on the 64 dev prompts. Push of a prompt = [logp(A) - logp(B)]
  probe on minus the same probe off. For each of the 64 heads alone, removed = (mean push - mean push with that head
  swapped) / mean push, a ratio of means over the 64 dev prompts (the parent's frozen metric, probe-switch-heads
  heads.py:14, :130).
- SET: every head with removed >= 0.05. If none, SET = the top-1 head. Then, while the t=0 SWAP dev strict of SET is
  >= 0.25, add the next head by audit rank; cap 32 heads.
- TOP-1: the head with the largest removed.
- LOO (leave one free): all 64 heads except the one with the smallest |removed| (ties: lower block, then lower index).
- ALL-64: all 64 heads.
- RANDOM (SANITY cell, named before data): |SET| heads drawn from the heads outside SET with
  random.Random(1117 + seed), draws 1 and 2, on the first two formed seeds. At 1.7B a random set removed nothing at t=0
  (128/128), so RANDOM is expected to test no return; the post does not call it a passed control.

### Clamp and retraining

- Clamp = SWAP for every arm: in the probe-on pass, each clamped head's slice is replaced by the detached slice of the
  same prompt's probe-off pass under the current weights, all positions. UMEAN (a fixed mean slice) is reported at t=0
  only, as a side row.
- Retraining: warm start from the seed's saved switch adapter, clamp on from update 0 in every training and scoring
  forward, the switch recipe above, dev strict every 64 updates. Budget = 2 x T_f of that seed (rounded up to a
  multiple of 64).
- RETURNED = dev strict >= 0.75 at 2 consecutive dev evals; return update = the update of the second; the run stops.
  Test strict (128 test prompts) is measured at the stop or at the budget.
- t=0 validity: if SET's t=0 SWAP TEST strict is >= 0.25 for a seed, that seed's SET cell reads "not removed at t=0"
  and counts as UNCONFIRMED for that seed.
- CONVERGED (the parent's rule, verbatim): the running mean of the training loss over the last 20% of updates vs the
  20% before differs by <= 0.02 nats, AND dev strict over the last 3 evals lies within a range of 0.10. A plateau at
  ln 2 counts.
- Run order (frozen; only changes what a time cut drops): SET x3, ALL-64 x3, TOP-1 x3, LOO x3, RANDOM 2 draws x 2 seeds.

### Title status (answers the question; the 1.7B rule)

Counted over the 3 formed seeds (both REJECTED and CONFIRMED need 3 formed seeds with a valid SET cell):
- REJECTED (no) = the SET arm RETURNED within the budget, or has test strict >= 0.25 at the budget, in >= 2 of 3 seeds.
- CONFIRMED (yes) = the SET arm has test strict < 0.25 at the budget with CONVERGED loss in >= 2 of 3 seeds.
- UNCONFIRMED otherwise (fewer than 3 formed seeds, SET not removed at t=0, non-convergence, cells missing at the
  wrapup).
- Headline seed-level quantity (proof-matrix.md): SET test strict at the stop or the budget, mean over the 3 seeds with
  a 95% t-interval (df 2, t = 4.303), next to the prompt-level interval per seed.

### Prereg outcome (the scout's prediction; reported apart from the status)

- held = SET RETURNED with return update <= T_f in >= 2 of 3 seeds, AND TOP-1 RETURNED within the budget in >= 2 of 3,
  AND LOO RETURNED within the budget in >= 2 of 3, AND ALL-64 test strict < 0.25 at the budget in >= 2 of 3.
- failed = SET not RETURNED by T_f in >= 2 of 3, OR TOP-1 or LOO held (not RETURNED) in >= 2 of 3, OR ALL-64 test
  strict >= 0.25 (or RETURNED) in >= 2 of 3.
- unsettled otherwise.

### The predicted number: return ratio

- r_s = SET return update / T_f, per seed s (SET RETURNED seeds only). Seed-level: mean of r_s with a 95% t-interval
  over seeds (df = n - 1; n = 3: t = 4.303).
- 1.7B band (procedure frozen, numbers not yet known): rerun the parent's SET arm on its three 1.7B switch adapters on
  the mini with this code at dev grid 64, after this push and before any 8B data; r_1.7B,s = return update / the
  parent's T_f (1,280 / 1,408 / 1,280, read on the parent's 128 grid; disclosed). Band = [min r - 64/T_f_min,
  max r + 64/T_f_min] with T_f_min the smallest of the three parent T_f, i.e. widened by one 64-update grid step on
  each side. Written to data/band_17b.json before any 8B retraining data exists. (For reference only, the published
  128-grid ratios: 512/1,280 = 0.40, 512/1,408 = 0.36, 384/1,280 = 0.30.)
- Reading (primary, by seeds; Vera #23554, Tove #23552): FASTER at 8B if >= 2 of 3 seed ratios lie below band low;
  SLOWER if >= 2 of 3 lie above band high; SIZE-INVARIANT if >= 2 of 3 lie inside the band; otherwise UNRESOLVED.
  A seed whose SET arm did not return within the budget counts as above the band.
- Reported next to it (not the reading): the seed-level t-interval read: FASTER if its upper end < band low, SLOWER if
  its lower end > band high, SIZE-INVARIANT if it lies inside the band, else UNRESOLVED. (Vera's simulation,
  skeptic/vera_ratio_power.py: with 3 seeds the interval read is UNRESOLVED most of the time.)
<!-- /FROZEN RULE -->

## Comparison note (written before data)

The 1.7B retraining budget was 4,096 updates = 2 x the parent's 2,048-update training = 2.9-3.2 x its T_f. Here the
budget is 2 x T_f (the goal's "up to 2x the first-learning updates"). Read at 2 x T_f, every 1.7B cell keeps its result:
every 1.7B return came by 1.455 x T_f (LOO s1, 2,048 / 1,408), and ALL-32's dev strict stayed <= 0.094 through
2 x T_f in all three seeds (head-clamp-retrain data/status.json). So the 1.7B rows in the figure are read at 2 x T_f.

## Side rows (reported, never status)

- P-COND (forward only, as 1.7B D5): conditional t=0 audit with SET clamped plus one more head; does the returned
  carrier sit in its top 3 (chance about 3 / (64 - |SET|))?
- WHERE audit of each RETURNED SET model (SET still clamped, one more head swapped, 64 dev prompts).
- UMEAN at t=0.

## Wrapup

If laptop jobs are still queued or running at 11:00 PST (20:00 Copenhagen), the room closes as
"UNCONFIRMED: waiting for laptop queue (<job ids>)" and the oracle continues the run from this branch.

## Priors (posted in the room before data)

Scout 0.60 (prediction holds). Tove ~0.7 (SET returns).
