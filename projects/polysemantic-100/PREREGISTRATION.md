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

Yes. Between 3 and 15 of the 100 neurons get the label UNRELATED (neuron label = worst pole label,
score.py). The swap null passes at most 5% of hypotheses. The pooled null gets UNRELATED on at least
80% of pairs.

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
