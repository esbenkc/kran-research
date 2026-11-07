# PRE-REGISTERED PREDICTION v2 — head-clamp-retrain (room theory-1107-near-tie-clamp)

Status of this file: v2 DRAFT. It becomes "v2 FROZEN" in the commit that fills the two gate blanks (PRIMARY clamp,
d79) from the forward-only t=0 gate, before any training update. Later changes go to DEVIATIONS.md with commit time.
Written by Iris (research) and Felix (plan) after Tove's mechanical VETO (#16215); reviewed by Kasper (#16232),
Vera (#16231), Tove (#16235). The scout's v1 (goal post #16196, fixed 14:52 CET) stays on record, unchanged.

## Title (yes/no)

Does clamping a set of attention heads chosen before retraining stop a learned switch from coming back?

- Reasonable: does clamping the audit's head set stop the switch from returning under retraining?
- Ambitious (answered, same compute): a head set read off ONE saved audit and fixed before retraining blocks the
  reroute for 2x the original budget, where one-head removal does not; an all-heads ceiling and a leave-one-free
  cell test whether only the whole layer holds.

## Scout v1 prediction (on record, unchanged; P = 0.40)

"In >= 2 of 3 seeds, the near-tie clamp keeps the switch below 25% of prompts after 2x the original training budget,
and the training loss has converged. The same-size random clamp lets it return to >= 75% in >= 2 of 3 seeds. The
clamp raises held-out LM loss by < 2%."

## D1: why v2 (disclosed before any data)

The v1 set rule ("every head in the last 2 layers within 5% of the top swap score") selects ONE head per saved seed
(`dev_single_cut_removed`, probe-switch-heads `data/L/heads`: s0 27.7 0.731 vs next 0.216; s1 27.7 0.915 vs 0.349;
s2 27.12 0.959 vs 0.475). Under v1 the near-tie clamp is one-head removal, the thing the ambitious claim contrasts
against. v2 keeps the claim and the thresholds and changes the set rule. The v1 set survives as the TOP-1 arm.

## Material (frozen)

- Seeds: the three label-trained switch models of probe-switch-heads (branch swarm/oracle-1024-probed-only-switch),
  Qwen3-1.7B bf16, LoRA r16 alpha 32 on q,k,v,o,gate,up,down of blocks 26-27, 2,048 updates:
  `data/L/runs/lora_none_s0_2048_adapter.pt`, `lora_none_s1_adapter.pt`, `lora_none_s2_adapter.pt`.
- First audit = saved field `dev_single_cut_removed` in `data/L/heads/<run>.json` (single-head swap, 64 dev prompts).
- Block-26 inputs (probe on / off; train 512, dev 64, test 128) rebuilt with the parent's chain_j1 method.
- Retraining: warm start from each seed's adapter, parent recipe (8 prompt pairs per update, micro 4, lr 1e-4 AdamW,
  clip 1.0, full-vocabulary CE at the answer position, probe on -> 'A', off -> 'B'), clamp on from update 0 in every
  training and scoring forward, 4,096 updates = 2 x 2,048.
- Head unit: one query head's 128-column slice of the o_proj input, all positions.
- Switch rate = strict switch on the 128 test prompts (probe-on top-1 'A' AND probe-off top-1 'B'), cached eager
  tail, never a recompiled whole graph. Also reported: reversed share n (probe-on not 'A' and probe-off 'A'), push removed.

## Clamp kind: mechanical t=0 gate (rule posted #16226 before the gate, amended by Kasper #16232 fix 2 before the gate)

- UMEAN: each clamped head's slice replaced by a POSITION-FREE constant: its mean over all positions >= 1 of the 512
  UNPROBED train prompts, and at position 0 (the attention-sink token) its own position-0 mean. Label-free, needs no
  second pass, applies to text of any length; the same two vectors in training, at t=0 and for LM loss (Kasper #16238).
- SWAP: in the probed pass each clamped head's slice replaced by the detached slice of the same prompt's unprobed
  pass under the current weights, all positions (the audit's own op; the parent head-free arm).
- Gate (forward only, no update): SET and TOP-1 arms x 3 seeds x {UMEAN, SWAP}, t=0 test strict. PRIMARY = UMEAN only
  if SET AND TOP-1 t=0 test strict < 0.25 on all 3 seeds; else SWAP. One kind for every arm, never mixed.
  Known before the gate: top-1 SWAP 0.023 / 0.000 / 0.016; top-1 both-conditions mean 0.961 / 0.273 / 0.172.
- PRIMARY = ______ (gate file `data/gate_t0.json`).
- Under SWAP the post says the clamp needs the probe-off pass, so it tests routing (who carries the signal), not a
  deployable defence (Tove #16224).

## Arms (head names frozen; script `felix_sets.py` -> `data/sets.json`, rebuilt independently by Vera `vera_draws.py`)

| arm | s0 | s1 | s2 |
|---|---|---|---|
| SET: every 26-27 head with single-cut removed >= 0.05 | 27.7, 26.5, 26.0, 26.14, 27.12, 27.3 | 27.7, 26.0, 26.2, 26.5 | 27.12, 26.0, 26.2 |
| TOP-1 (v1 literal rule) | 27.7 | 27.7 | 27.12 |
| RANDOM same size from 26-27 outside SET, draw 1 of `random.Random(1107+seed)` | 26.6, 26.12, 27.0, 27.4, 27.10, 27.14 | 26.10, 27.4, 27.8, 27.12 | 26.4, 26.14, 27.1 |
| ALL-32 (ceiling) | all 32 | all 32 | all 32 |
| LEAVE-ONE-FREE: all 32 except the head with the smallest abs dev share (ties: lower block, then index) | free 26.11 | free 26.7 | free 27.2 |

Random draws 2-3 (only if time): s0 [26.1, 26.3, 26.7, 26.13, 27.0, 27.10], [26.6, 26.10, 26.11, 26.15, 27.6, 27.9];
s1 [26.12, 27.1, 27.10, 27.14], [26.8, 27.5, 27.13, 27.14]; s2 [26.4, 27.5, 27.6], [27.3, 27.13, 27.14].
The s0 SET is threshold-sensitive: 27.3 = 0.0506 and 27.12 = 0.0508 sit on the line; 26.4 = 0.0445 just under.
RANDOM (Kasper fix 4): under SWAP a random set outside SET removes little at t=0 (random_ex_cut mean 0.07 on s0),
so 'RANDOM >= 0.75' is met before training. It is a sanity cell, not evidence; the post does not call it a passed control.

## Rules (frozen)

- Dev strict every 128 updates and at t=0. RETURNED = dev strict >= 0.75 on 2 consecutive evals; the run stops there,
  test strict is measured at the stop. Formation/return time = the update of the SECOND eval of the pair (Vera #16234).
- CONVERGED (Vera #16212): running mean of train loss over the last 20% of updates vs the 20% before, |diff| <= 0.02
  nats, AND dev strict over the last 3 checkpoints within 0.10. A plateau at ln 2 counts.
- TITLE STATUS: CONFIRMED (yes) = SET arm test strict < 0.25 at 4,096 updates with CONVERGED loss in >= 2 of 3 seeds.
  REJECTED (no) = SET arm test strict >= 0.25 (or RETURNED) in >= 2 of 3 seeds. Otherwise UNCONFIRMED (non-convergence,
  or the primary clamp leaves the SET arm >= 0.25 at t=0). The TLDR states the returned rate per seed.
- PREREG OUTCOME (scout clauses with SET for the near-tie set): held = SET < 0.25 converged in >= 2/3 AND RANDOM
  >= 0.75 in >= 2/3 AND LM loss rise < 2%; failed = SET >= 0.25 in >= 2/3, or RANDOM < 0.75 in >= 2/3, or LM >= 2%;
  unsettled otherwise. Under SWAP the LM clause is true by construction on probe-off text (D2): reported as such,
  not as a measured pass.

## Interpretation table (Tove #16235; frozen so the story is not picked after data)

| case | reading |
|---|---|
| (a) SET holds AND TOP-1 returns (>= 0.25 in >= 2/3) | ambitious YES: a set fixed from one audit blocks the reroute that one-head removal cannot |
| (b) SET holds AND TOP-1 holds | YES to the title, headline 'one head was already enough here'; set-vs-single NOT licensed, the post says so |
| (c) SET returns, ALL-32 holds | NO: the reroute uses heads outside the audit set (WHERE names them) |
| (d) SET returns AND ALL-32 returns | NO, plus the d79 / acc_f reading: the answer-position MLPs suffice |
| (e) LEAVE-ONE-FREE returns while ALL-32 holds | impossibility line: one head the audit scored ~0 carries it back; reported with any of (a)-(d) |

## Numbers predicted before data

- Priors posted before v2 (P that SET stays < 0.25 in >= 2/3): Vera ~0.2, Kasper ~0.2, Bruno ~0.15, Felix ~0.25,
  Tove 0.35-0.45, Iris 0.30. Scout v1 P = 0.40.
- ALL-32 (Kasper #16219, Vera #16221): with every 26-27 head probe-independent, the answer logits are a function of
  r79 (block-26 input at position 79) alone. Exact identity s = 2 acc_f - 1 + n (s strict, n reversed share, acc_f
  the item accuracy of the clamped tail's own answer rule). Hard bound: s <= d79 = share of test prompts whose cached
  r79 differs in any bit between probe on and off. d79 = ______ (s0 / s1 / s2 share one cache; from the rebuilt cache
  before training). Soft estimate: s ~ 2(0.52) - 1 + n = 0.04 + n (logistic accuracy at r79 from the parent map; an
  estimate of the best rule, not a bound). Report n per seed.
- LEAVE-ONE-FREE (Iris #16217): on s0 the free head carries the switch back (test strict >= 0.25 within 4,096). P = 0.55.
  Reason: the answer position reads at chance (AUROC 0.52), position 71 reads 0.90, so a route must cross positions in
  a 26-27 head, and LoRA trains q,k,v,o of the free head.
- WHERE (Kasper #16211, fix 6): applies only if the s0 SET arm RETURNS. A single-head swap audit of the returned model
  (32 heads, the same 64 dev prompts, SET still clamped, ranking by the same 'removed' field) has at least one of
  26.2, 26.15, 26.3 in its top 3.
- RETURN TIME (Iris #16233, Vera #16234): the switch first formed (second eval of the first pair >= 0.75) at 1,280 /
  1,408 / 1,280 updates from scratch (s0/s1/s2). Any arm that RETURNS does so at or before its seed's formation time.

## LM loss (Kasper #16228/#16238, fix 5)

Held-out text: NeelNanda/pile-10k, first 64 documents, one 128-token window each (block-26 inputs from the base
model's untouched layers 0-25, probe off). LM loss = mean next-token CE of the tail on the saved switched adapter of
each seed, UMEAN clamp vs no clamp on the same weights, for SET and ALL-32. Logged in the gate; it does not enter the
clamp selection. Under SWAP, LM loss on probe-off text is unchanged by construction (D2): reported as such.

## Compute (for Vera's gate; one heavy job at a time, nothing in parallel)

| order | job | est. peak RSS | time |
|---|---|---|---|
| 1 | cache rebuild (chain_j1 method, full model, 704 prompts, probe on/off; SHA256 must equal the parent manifest) + d79, then pile-10k block-26 states | ~5.4 GB | ~4 min |
| 2 | t=0 gate: blocks 26-27 tail, 3 seeds x {SET, TOP-1} x {UMEAN, SWAP} + LM loss | ~3.2 GB | ~5 min |
| 3.. | training runs, one at a time | ~3.2 GB | <= ~22 min each at 4,096; returned runs stop early |

Training order: s0 SET, s0 TOP-1, s0 ALL-32, s0 LEAVE-ONE-FREE, s0 RANDOM; s1 SET, s1 TOP-1; s2 SET, s2 TOP-1;
s1/s2 RANDOM; s1/s2 ALL-32; s1/s2 LEAVE-ONE-FREE; random draws 2-3. Then the WHERE audit for each returned SET run.
Time cut (Kasper #16253): training ends by 09:30 PST (18:30 CET); drop from the end of the order. SET and TOP-1 x 3
seeds are never dropped.
ENGINE REPLICATION (Kasper #16253): s0 ALL-32 under SWAP repeats the parent's head-free cell (probe-switch-audit-penalty
hf_s0.json: same warm adapter sha256 56fbff28, same swap op). Its dev curve must match hf_s0 (dev strict <= 0.11, train
CE ~ ln 2) at 512 and 2,048 updates; if not, the engine is wrong and every arm is suspect (stop and fix before more runs).
If PRIMARY = UMEAN, s0 ALL-32 also runs once under SWAP as this check.
