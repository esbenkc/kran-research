# Pre-registration — polysemantic-100

Fixed: October 18, 2026, 04:33 Copenhagen time (October 17, 19:33 PDT). Written by Iris (research)
from Selma's draft. At this time the record pass was running (about 320 of 1,590 windows). No
context extraction, hypothesis, probe or judge output existed (`data/` held only acts.f16, gate.f16,
tokens.npy, corpus.json, neurons.json, logs). Nobody had read any activation value.

Method, sample and rules are fixed in earlier commits: 75c08c4 (seed 20261017, 100 neurons,
corpus), a15b9bc (context extraction), c8e3723 (hypothesis prompt, probe pass rule, judge prompt,
labels, nulls), 70c8498 (judge models), 10d9c8d (ablation rule).

## Question

If we draw 100 neurons of Qwen3-4B-Instruct-2507 (4-bit) at random, will any of them do two
unrelated jobs that each pass a minimal-pair test?

## Expected

Yes. Between 3 and 15 of the 100 neurons get the label UNRELATED under rule 1 (neuron label = worst
pole label, score.py). The swap null passes at most 5% of hypotheses. The pooled null gets UNRELATED
on at least 80% of pairs. The split null gets RELATED on at least 80% of pairs. At least one
UNRELATED neuron passes ablation for both jobs.

## Confidence

60%.

Basis: Goodfire's own judge scores 25 of 500 random Qwen3-32B neurons at 3 or less out of 10 for
clean (5.0%) and 52 of 500 at 4 or less (10.4%) (gallery data, research.md section 2). Qwen3-4B has
about a fifth as many MLP neurons, and Goodfire's GPT-2 XL redo found more mixed neurons in a small
model, so we expect at least as many. Our probe gate is strict (3 of 4 fire texts at half the
reference max, 3 of 4 controls under half), and hypotheses come only from 407k corpus tokens, so the
count is a lower bound. These two pulls make the range wide.

## If right

Unrelated jobs in one neuron are common enough to find in a random sample of 100, at a rate near
Goodfire's 5% for Qwen3-32B.

## If wrong

Below 3: unrelated jobs are rarer than Goodfire's own judge suggests, or our test misses them.
Above 15: a small model packs more unrelated jobs into one neuron, or our probes are too loose
(check the swap null first).

## Rules added after Kasper's review (fixed October 18, 2026, 04:35 Copenhagen time, before any probe, judge or ablation output exists; the Expected section gained the split-null and ablation clauses at the same time)

1. **Probe reference max.** Primary rule = the score.py docstring: ref = max(corpus pole max, every
   probe text of this pole, over all its hypotheses). Sensitivity rule (reported, never used for the
   headline) = the score.py code at c8e3723: ref = max(corpus pole max, the 8 texts of this
   hypothesis). score.py must use the primary rule for `labels.json`.
2. **Split null.** The judge must see two descriptions written independently: for each passed
   hypothesis with at least 4 covered snippets (at most 30), the explain prompt runs once on each
   half of the covered snippets, and the judge compares the two new descriptions. Bar: at least 80%
   RELATED. (The c8e3723 version showed the same description twice and could not fail.)
3. **Ablation baseline.** For each fire text, the baseline neurons are the 20 other neurons of the
   same layer with the largest |activation| at the target's peak token on that text (activation-
   matched), each zeroed alone. A job's ablation passes if the target's effect is larger than all 20
   (one-sided p < 0.05). The 20 random same-layer neurons of 10d9c8d are reported as a second, weaker
   baseline ("matters more than a typical neuron").
4. **Title answer.** Yes = at least 1 neuron is UNRELATED under rule 1 AND the swap null passes at
   most 5% AND the pooled null gets at least 80% UNRELATED AND the split null at least 80% RELATED AND
   that neuron's ablation passes for both of its jobs (rule 3). If some UNRELATED neuron exists but
   a null misses its bar or no UNRELATED neuron passes ablation for both jobs: Not settled. If no
   neuron is UNRELATED under rule 1 and all nulls meet their bars: No.
