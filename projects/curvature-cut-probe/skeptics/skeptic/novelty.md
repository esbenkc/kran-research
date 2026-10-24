# Vera — novelty check (own searches, 2026-10-24)

Claim to check: "restricting (refitting) a falsehood probe to the top output-Fisher directions of the last
post-norm layer changes what an output-matched fine-tune pays to hide from it", with a pre-computed floor.

## What is NOT new (credit)
- The floor itself. FishBack (Sihan Wang, Jiayi Zhao et al., arXiv 2605.17231, "Pullback Fisher Geometry for Optimal
  Activation Steering in Transformers", May 2026): closed-form minimum-KL steering direction for a target concept
  change under the pulled-back output Fisher. Our second-order floor D^2/(2 a'F^+a) is that result read from the
  attacker's side. Checked: https://arxiv.org/abs/2605.17231v1 (abstract states the closed-form minimum off-target KL
  direction "via a single matrix solve").
- Probes fall to output-matched obfuscation: Bailey et al. arXiv 2412.09565 (Obfuscated Activations), RL-Obfuscation
  arXiv 2506.14261, Neural Chameleons. None of them changes the probe's geometry to raise the price.

## Searches (web_search, 2026-10-24)
1. "probe restricted to output Fisher subspace evasion cost KL obfuscation ..." -> no usable hit.
2. "FishBack Fisher pullback concept steering minimum output movement" -> FishBack only (steering, not monitoring).
3. "linear probe restricted to unembedding row space robust to obfuscation attacks deception probe output-relevant
   subspace" -> 2412.09565, 2506.14261, 2605.27958 (pressure-testing deception probes), 2610.27577 (static probes);
   none restricts a monitor to the output-sensitive subspace and measures the attacker's price against a floor.

## Added after Hugo's check (2026-10-24)
- Zhong, "Spurious probes as a black-box alternative to activation probing", LessWrong 2026-09-25: readouts through output tokens
  held up better against a Neural Chameleon than activation probes. Same spirit ("read what the outputs depend on"), different
  design (black-box questions, no output-matched fine-tune, no price). Credit line needed.
- Neural Chameleons: McGuinness, Serrano, Bailey, Emmons, arXiv 2512.11949 (attack shape: lambda x KL to base + evasion term).

## Verdict
No prior work found that (a) refits a falsehood probe inside the top output-Fisher subspace and (b) attacks it with an
output-matched fine-tune to compare the measured price with the pre-computed floor ratio. The floor and the attack
recipe are prior work and must be credited; the test of the design rule is new as far as these searches reach.
Scope limit: 3 web searches + Iris's 13 full texts (research.md); not a systematic review.
