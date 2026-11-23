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

## Delta, oracle-1123-audit-penalty-8b (Vera, 2026-11-22 PST)

Search (independent of Iris): "adversarial training evade activation patching circuit discovery audit attention heads
backdoor 2026" (web, last month). New hits since 2026-11-20 and why they are not the same experiment:
- Circuit Condensation: Post-Training that Concentrates a Behavior's Causal Circuit, arXiv 2608.27254: trains so a
  behaviour's causal circuit gets SMALLER (the opposite direction); no trigger-conditioned switch, no audit penalty.
- DetectViT, arXiv 2611.23770 (2026-11-18): test-time backdoor detection from attention-head discrepancy in vision
  transformers; no attacker trains against it.
- Backdoor Attribution (BAHA), ICML 2026 poster (= arXiv 2509.21761, already listed): ~3% of heads carry a backdoor;
  our L0 control agrees in kind (one head removes 0.75 of the push), our AO arm shows that a trained attacker can push
  the largest single-head share down to 0.14-0.20 but not the joint top-3 share below 0.49.
Iris's additions (Gong et al. 2607.01940 dormant backups; Gupta et al. 2611.07891 larger models compensate less;
McGrath et al. 2307.15771 hydra effect) are priors for the mechanism and the scale reading, not the same experiment.

Verdict unchanged: novel as an experiment (gradient attacker vs a head-level patching audit, at 8B, with an untrained
-penalty control). No record table applies; the comparison is our own 1.7B run.
