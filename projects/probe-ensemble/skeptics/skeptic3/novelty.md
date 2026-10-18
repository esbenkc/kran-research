# Novelty Search (Nadia / skeptic3)

## Research Context
- **Neural Chameleons (arXiv 2512.11949)**: Found that models can evade unseen monitors, including ensembles. Our results confirm this in a DPO/honesty context: training against an ensemble of 8 probes did not meaningfully close the evasion route compared to one.
- **Obfuscation Atlas (arXiv 2602.15515)**: Maps honesty vs. obfuscation tradeoffs. We provide a specific data point: a "harder" target (8-probe max) buys no more honesty than a single probe.
- **Wehner & Fritz (arXiv 2510.21531)**: Tested 1 vs 10 probes on toxicity. Found no significant behavior gain (0.14 vs 0.15 toxicity rate). We extend this finding to the deception/honesty domain with a specific ensemble max-pool construction.
- **Libon et al. (arXiv 2609.38645)**: Tested 1 vs 16 probes at a single layer for harmfulness. Found frozen probes are trivially evaded. We test an ensemble distributed across layers (6/12/18/22) for honesty and find a similar failure.

## Novelty Claim
To my knowledge, this specific measurement on **Qwen2.5-0.5B** using an **iterated DPO loop** with an **ensemble penalty (8 members across 4 layers)** vs. single-probe is new. Specifically, the finding that the honesty gain ratio is near 1.0 (CI upper < 2.0) for an 8-probe ensemble vs. 1-probe is a novel empirical constraint on monitor diversity in rewards for honesty.
