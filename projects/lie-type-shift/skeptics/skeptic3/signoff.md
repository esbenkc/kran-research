# skeptic3 sign-off — lie-type-shift

**Verdict: SIGNED OFF on FAILED (lab_status "NULL").** The pre-registered prediction did not hold
under either judge. My own code reproduces build's numbers from the saved labels.

API spend by skeptic3: $0 (no model calls; stdlib Python on saved files).

## What I rebuilt (own code, reads only saved files)

Inputs: `projects/lie-type-shift/data/labels/labels_{gpt-4.1-mini,claude-sonnet-4-6}.jsonl`
(4,000 rows each), `data/handcheck/{sample_key.jsonl,labels_agent.csv}`, and the original
probe-goodhart eval rows (`eval_lam{0,1}_s{0,1}.jsonl`, for `n_words` and the old binary judge).

- `rebuild.py` → `out/rebuild.json`: M = (OMISSION + MISLEADING_TRUE) / (FALSIFICATION + OMISSION +
  MISLEADING_TRUE), seeds pooled, Delta = M(r8) − M(r0), 2,000 prompt-clustered bootstrap
  resamples; two null tests; length confound; judge agreement.
- `lexical_proxy.py` → `out/lexical_proxy_binaryjudge.json`: judge-free omission proxy.
- `hand60.py` → `out/hand60.json`: per-class scoring of both judges against the blind agent labels.
- `old_vs_new.py` → `out/old_vs_new.json`: old binary judge vs new rubric drop (see retraction).
- `sens_olddec.py` → `out/sens_olddec.json`: M on answers the old judge also called deceptive.

## Key numbers (match build to the digit)

| judge | Delta lam=1 [95% CI] | Delta lam=0 [95% CI] | rule |
|---|---|---|---|
| gpt-4.1-mini | +2.9 pp [−1.4, +7.7] | −13.6 pp [−19.2, −8.2] | both conditions fail |
| claude-sonnet-4-6 | −0.1 pp [−7.6, +7.2] | −26.1 pp [−34.5, −18.9] | both conditions fail |

Per seed (lam=1): mini +2.7 / +3.1; sonnet +1.9 / −2.1. lam=0: mini −20.6 / −6.2; sonnet −32.7 / −19.4.
Round-0 labels are identical across the four run copies for both judges (checked).

## Null tests

1. Round permutation within (prompt, seed): lam=1 Delta p = 0.07 (mini), 0.98 (sonnet); lam=0 p
   < 0.001 both. The penalised arm's Delta is indistinguishable from no change.
2. Arm swap at round 8 within (prompt, seed), diff-in-diff Delta(lam1) − Delta(lam0): +16.5 pp
   (mini), +26.1 pp (sonnet), p < 0.001 both. The arms differ, but because the *control* moves
   toward falsification, not because the penalised arm moves toward omission.

## Confounds checked

- **Length.** Median answer length swings across rounds in both arms. M by length tercile is flat
  (sonnet 0.63 / 0.62 / 0.64), and reweighting round 8 to round 0's length mix leaves the verdict
  unchanged (sonnet lam=1 M_r8 0.68 → 0.65 len-matched; lam=0 0.42 → 0.30).
- **Judge-free proxy.** Share of old-judge-deceptive answers with zero word overlap with the
  sensitive fact (pure omission): lam=1 18.3% → 4.0% (−14.3 pp, CI [−20.9, −7.8]); lam=0
  18.3% → 13.9% (CI [−11.2, +2.5]). The penalised model names the hidden fact *more* often. This
  agrees with both judges' OMISSION share falling 40% → 10% of answers in lam=1.
- **Honest contamination.** Both judges call most hand-HONEST answers MISLEADING_TRUE (HONEST
  recall mini 3/15, sonnet 2/15). This inflates M where honest answers are common (lam=1 late
  rounds), i.e. it biases toward the prediction. Removing it (M only on answers the old binary
  judge also called deceptive; hand agrees with that judge 55/60): sonnet lam=1 −10.6 pp
  [−19.2, −1.4], lam=0 −27.6 pp [−36.0, −19.7]; mini +1.9 / −13.8. Still FAILED; with sonnet the
  sign points the other way (toward outright false statements, slower than control).
- **Judge validity on the metric's key split.** Within answers both hand and judge call deceptive,
  false-vs-no-false accuracy: mini 52% (n=44, ~chance; finds 4/25 hand-FALSIFICATION), sonnet 84%.
  Mini's base M = 0.93 is mostly its blindness to falsification, so it had no room to rise 15 pp.
  The verdict should be carried by sonnet. Caveat: the hand labeller is a Claude-family agent, same
  family as sonnet.

## Retraction (logged in room #1366)

In room #1354 I said most of the old judge's lam=1 "lying drop" (25.7 pp) vanishes under the new
rubric (9.8 / 5.7 pp) and is minimisation. The hand check contradicts it: on the 11 sampled
answers the old judge called HONEST, the hand label is HONEST 11/11 and I read ~8–9 of them as
plain disclosures. The gap is mostly the new judges' over-strict HONEST line. **Do not put that gap
in the post as a finding.** At most one sentence in limits.

## Conditions on this sign-off

1. The post says the mini judge cannot separate false from true-but-misleading here, and leans on
   sonnet.
2. The post does not adopt the prereg "If not" text ("the lying drop is real honesty") as a
   conclusion; it is not tested by this run.
3. The "omission → misleading truth" pattern is phrased as "the penalised model stays silent less
   and names the hidden fact more" (judges + judge-free proxy agree), not as hidden minimisation.
4. Hollinsworth et al. 2026 (arXiv 2607.01567) is cited as the closest prior observation (see
   `novelty.md`).
