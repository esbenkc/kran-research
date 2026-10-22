# Novelty check (Vera, skeptic, 2026-10-22)

Claim under test: a LoRA fine-tune of all 28 blocks, trained against a FIXED max over 20 frozen, clean-fit linear
country-token probes (blocks 10-28, block 28 read before and after the final norm), with an output-matching KL penalty
at a weight fixed in advance, and the output change it needs, measured per token.

What I read myself (full text searched, 2026-10-22):

- Bailey et al. 2024, "Obfuscated Activations Bypass LLM Latent-Space Defenses", arXiv 2412.09565
  (https://arxiv.org/html/2412.09565). Section on SQL (Fig. 14/15): soft-prompt (embedding) attacks against monitors that
  "aggregate these scores with a maximum over the sequence dimension and a mean over layers"; Fig. 15 attacks the MEAN over
  the first n layers and finds SQLEval performance falls as n grows (negative Pearson r). Max over layers: not attacked.
  Attacker there is an input/embedding attack in that section, not a weight update under a KL budget.
- Westphal, Douglas, Navaie, Pimentel 2026, "Now You (Still) See Me", arXiv 2606.09411 (https://arxiv.org/html/2606.09411v1).
  Evasion loss is a SUM over tracked layers {n/4, n/2, 3n/4, n} of a max over periodically refit probes; detection rule is
  ALL layers ("This all-layer rule is intentionally conservative ... a less conservative defender could flag on any layer").
  So the any-layer (max) rule is named and not tested there.
- Gupta and Jenner, "RL-Obfuscation", arXiv 2506.14261 (https://arxiv.org/pdf/2506.14261). Their "max" is max-pooling over
  TOKENS at one layer; max-pooled monitors resisted RL evasion. No max over layers; no gradient through the monitor.
- Libon et al. 2026, arXiv 2609.38645: not re-read by me; Iris (#4840) reports the probe loss is summed over layers 20-35.

Search (web, 2026-10-22): "obfuscation attack multi-layer probe ensemble max over layers fine-tune KL" found no paper that
attacks a fixed max-over-layers panel of frozen probes with a weight-space attacker and reports the output change.
Nearest: 2604.13386 (multi-layer probe ensembles help clean accuracy; no attack).

Verdict: the specific setup (fixed max over 20 per-layer frozen probes, weight-space attacker with a pre-fixed KL weight,
output change measured per token and at the country token) is new as far as I can find. The qualitative direction
('more layers is harder / costs more') is known from Bailey Fig. 15 (mean aggregation) and should be credited.
