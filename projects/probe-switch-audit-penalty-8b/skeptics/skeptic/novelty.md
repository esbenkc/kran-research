role: skeptic
# Novelty check (Vera, oracle-1121-audit-penalty-8b)

Claim checked: no public paper trains a model (with gradient descent) against a head-level activation-patching
audit and reports whether a trigger-conditioned behaviour hides from that audit; and no public paper does this at
two model sizes (Qwen3-1.7B, Qwen3-8B).

Searches (2026-11-20, independent of Iris's research.md):
1. "training a model to evade activation patching attention head attribution audit backdoor penalty"
2. "obfuscated circuits adversarially robust to mechanistic interpretability causal scrubbing evasion fine-tuning"

Closest hits and why they are not the same experiment:
- Backdoor Attribution (BAHA), arXiv 2509.21761: locates backdoor heads by ablation/patching; no attacker trains
  against that attribution.
- Mechanistic Exploration of Backdoored LLM Attention Patterns, arXiv 2508.15847: patches heads of poisoned models
  (~24-31 heads to remove a trigger's effect); no adaptive training against the patching score.
- Mechanistic Interpretability in the Presence of Architectural Obfuscation, arXiv 2506.18053: weight/representation
  obfuscation maps, not a loss that penalises a head audit's score.
- Inherited Circuits, Learned Semantics (fine-tuning evasion), arXiv 2606.27091: evasion of evaluations after
  fine-tuning, not a head-audit penalty.
- Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 (cited by Iris): trains against
  probes / latent monitors, not a head-attribution audit; the nearest method family.
- Transformer Circuit Faithfulness Metrics are not Robust, arXiv 2407.08734: shows circuit scores depend on
  ablation choices; no adversarial training.

Prior result this post must be compared with: our own 1.7B run (blog.kran.ai/probe-switch-audit-penalty,
REJECTED: no seed at any weight fired >= 75% strict while the blind top-3 removed < 30%). The 8B run is a scale
replication plus one new arm (audit-only ratio penalty); the new arm has no published counterpart.

Verdict: novel as an experiment; the 8B result does not "beat" a published number, so no record table applies.
