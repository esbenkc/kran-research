# Vera novelty check (own searches, 2026-10-27 07:1x PDT)

Claim to test: "an output-matched (KL-anchored) fine-tune against linear probes that are refit with labels on its own
activations, with the output price predicted before training by closed-form LEACE erasure".

Checked sources (arXiv export API abstracts read by me today; ids resolve):

| work | what it did | overlap with us |
|---|---|---|
| Belrose et al. 2023, arXiv 2306.03819 (LEACE) | closed-form least-squares linear eraser; "concept scrubbing" at every layer; reports downstream effects | gives our price formula; does not compare to a trained evader's cost |
| Westphal et al. 2026, arXiv 2606.09411 "Now You (Still) See Me" | adversarial fine-tune of stego trojans (8-14B) evades a linear probe refit during training, and a held-out MLP probe; KL anchor to predecessor (Kasper read sec 4.2) | labelled refit probe + output anchor already attacked; planted payload, not a native concept; no closed-form price |
| Libon et al. 2026, arXiv 2609.38645 | training against continuously updated probes; abstract: "the concepts stay linearly encoded after fine-tuning" | updated probes, but no output matching (behaviour may change); opposite-leaning claim |
| Keenan et al. 2026, arXiv 2609.36490 | models learn to edit activations to evade latent monitors from feedback | monitors not refit with labels on the evaded states in the main attack (Kasper: App C.1 edit-aware probe recovers 0.986 TPR) |
| Bailey et al. 2024, arXiv 2412.09565 | obfuscated activations; weight-space backdoor attack with retrained monitors | defender has no labels on the evaded distribution |
| Elazar & Goldberg 2018, arXiv 1808.06640 | adversarial removal in encoders: a post-hoc probe recovers the attribute | old precedent for "refit recovers" |

Verdict: NOT new = "a fine-tune can beat a probe refit with labels under an output anchor" (Westphal 2026).
New as far as I found = (a) a closed-form price (LEACE KL, computed on the clean model before training) tested
against the measured output price of the trained evader, (b) on a concept the model already holds (truth of
city-country statements) and its outputs barely use. The post must not say "first" about the attack itself.
