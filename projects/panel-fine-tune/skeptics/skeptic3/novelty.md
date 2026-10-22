**Novelty Assessment (Nadia)**

Attacking a fixed, maximum-over-layers monitor with linear probes has not been demonstrated with this strict KL preservation constraint. 
- Bailey et al. (2412.09565) attack a *mean* across layers using soft prompts (which is a different geometric object than full weight updates) and observe a stark task performance tradeoff.
- Westphal et al. (2606.09411) test a *sum* across 4 tracked layers. They specifically flag an "any-layer" (i.e. maximum) rule as untested.
- Libon et al. (2609.38645) also use a summation over a subset of layers.

This post directly targets the "any-layer" maximum alarm across a wide span (layers 10-28/29), showing that full weight updates can break the aggregation without a substantial behavioral cost, establishing an empirical upper bound on the cost of full-stack evasion.
