# Would a Random Network Pass the "One Main Job" Neuron Audit?

**Question:** Does a blind LLM audit that asks "does this neuron have one main job?" (the kind behind Cammarata's
~90% estimate for Qwen3-32B in "Neural Networks Are Already Clean", Goodfire 2026) pass units of a randomly
initialized network, or random directions in a trained one, as often as real trained neurons?
**Sources:** Qwen/Qwen3-0.6B (Hugging Face); NeelNanda/pile-10k (public Pile subset); judges gpt-4.1-mini and
claude-haiku-4-5 via API. Pre-registration: PREREGISTRATION.md (frozen 2026-10-15 23:16 CPH).
**Method:** 450 units (150 per arm, 50 at each of layers 4/14/24): trained MLP neurons, random-init MLP neurons, and
random unit directions over z-scored trained neurons. Top-20 excerpts (max one per document) from 5M tokens; one judge
labels, the other family must pick 5 held-out top excerpts out of 10. Reproduce: `run.py`.

## Findings

One main job (pre-registered rule): trained neurons 14.3% [10.0, 19.0], random directions 1.7% [0.3, 3.3], random
network 0.0% (0 of 150). All four pre-registered hypotheses failed (non-lexical gap A-B 1.7 pp vs predicted 25 pp;
judge type labels unreliable, agreement 60%). Judge kappa 0.34; both judges agree on a pass for 6% of trained neurons.
Numbers: results.json; skeptic rebuilds: skeptics/.

## Blog post

Slug: `blog.kran.ai/_notes/random-network-neuron-audit.md`, figure id `random-network-neuron-audit`.
