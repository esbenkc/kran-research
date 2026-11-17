# PRE-REGISTRATION — head-clamp-retrain-8b (room theory-1117-head-clamp-8b)

Scale replication of head-clamp-retrain (Qwen3-1.7B, room theory-1107-near-tie-clamp, branch
swarm/theory-1107-near-tie-clamp, published REJECTED at https://blog.kran.ai/head-clamp-retrain/) on Qwen3-8B.
Written by Iris (research) from Felix's final plan (room #23533, #23535) with the fixes of Tove (#23525, #23536),
Kasper (#23529, #23540, #23541) and Vera (#23520, #23521). Frozen before any 8B data and before the 1.7B 64-grid rerun.
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
- Seeds: switch models s0, s1, s2 (main arms: 3 seeds). RANDOM control: 2 draws x seeds s0, s1 (2 seeds per control).

<!-- FROZEN RULE -->
### G0: the trace at our answer position (forward only, step 0 of the first laptop job)

- d34 = share of the 128 test prompts whose cached block-34 input at the answer position differs in any bit between
  probe on and probe off. If d34 < 0.5, the run stops; the post's first line is "the probe left no trace at the answer
  position at 8B (d34 = <value>)", status UNCONFIRMED.
- ALL-64 forecast (reported, not a status input): held-out accuracy of a logistic readout of probe on/off from the
  block-34 input at the answer position (train split fit, test split score; 1.7B analogue 0.535). Frozen reading:
  accuracy < 0.60 predicts that ALL-64 holds; >= 0.60 predicts that it returns.

### Switch training (from the base model)

- LoRA rank 16, alpha 32, on q, k, v, o, gate, up, down of blocks 34-35; the parent recipe: 8 prompt pairs per update
  (micro-batch 4), AdamW lr 1e-4, gradient clip 1.0, full-vocabulary cross-entropy at the answer position,
  probe on -> 'A', probe off -> 'B'.
- Strict switch = probe-on top-1 'A' AND probe-off top-1 'B' on the same prompt. Dev strict on the 64 dev prompts
  every 64 updates.
- T_f (first-learning update count) = the update of the SECOND of the first two consecutive dev evals with dev strict
  >= 0.75. Training continues to the first multiple of 64 >= 1.5 x T_f, then the adapter is saved. Cap: 4,096 updates.
- FORMED = T_f exists by 4,096 AND the saved adapter gives test strict >= 0.75 (128 test prompts). If fewer than 2 of 3
  seeds form, the run stops after the switch step; the post's first line is "the switch did not form within 4,096
  updates at 8B in <k> of 3 seeds", status UNCONFIRMED. A seed that does not form is excluded from every later arm.

### Audit and head sets (picked by code inside the laptop job; no human sees the audit before the sets exist)

- Single-head SWAP audit of each formed switch model on the 64 dev prompts: for each of the 64 heads alone, 'removed'
  = the share of the probe's push (mean over prompts of [logp(A) - logp(B)] probe on minus probe off) that the swap
  removes.
- SET: every head with removed >= 0.05. If none, SET = the top-1 head. Then, while the t=0 SWAP dev strict of SET is
  >= 0.25, add the next head by audit rank; cap 32 heads.
- TOP-1: the head with the largest removed.
- LOO (leave one free): all 64 heads except the one with the smallest |removed| (ties: lower block, then lower index).
- ALL-64: all 64 heads.
- RANDOM (SANITY cell, named before data): |SET| heads drawn from the heads outside SET with
  random.Random(1117 + seed), draws 1 and 2, on seeds s0 and s1. At 1.7B a random set removed nothing at t=0
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

- REJECTED (no) = the SET arm RETURNED within the budget, or has test strict >= 0.25 at the budget, in >= 2 of 3 seeds.
- CONFIRMED (yes) = the SET arm has test strict < 0.25 at the budget with CONVERGED loss in >= 2 of 3 seeds.
- UNCONFIRMED otherwise (a seed not formed, SET not removed at t=0, non-convergence, cells missing at the wrapup).

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
- Reading: FASTER at 8B if the 8B interval's upper end < band low; SLOWER if its lower end > band high; SIZE-INVARIANT
  if the interval lies inside the band; otherwise UNRESOLVED.
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
