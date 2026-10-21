# Hugo / skeptic2 novelty notes

Question checked: Does adding back the base refusal direction at one layer restore a stripped model's refusals without over-refusing?

Sources checked:
- Arditi et al., `arXiv:2406.11717`, "Refusal in Language Models Is Mediated by a Single Direction". The abstract states that erasing one direction prevents refusal and adding the direction elicits refusal on even harmless instructions. This makes harmful-only restoration and harmless over-refusal expected, not new.
- Lee et al., `arXiv:2610.00320`, "Refusal Localizes, the Damage Relocates". Section 4.1 reports frozen clean probes remain separable after attack: AUROC 0.892-0.997 across Llama cells and 0.750-0.995 over 72 cells across six checkpoints. Appendix H, "Writing the direction back", swept clean refusal direction add-back over contiguous layer bands, four gains, and two write rules. Best Llama arm: 0.37 refusal vs clean 0.92. One Qwen arm: 0.98 vs clean 0.99 on harmful prompts, but 0.66 benign refusal vs clean 0.00. Therefore the exact harmful-only add-back claim is already partly shown, and the novelty is the single-layer selective frontier at a strict benign-refusal budget.
- Glass et al., `arXiv:2608.24988`, "Does Fine-Tuning Undo Activation Steering?". The abstract reports fine-tuning can recover behavior without reversing a weight edit; it studies durability of embedded steering under later fine-tuning, not post-hoc re-adding the base refusal direction to a stripped model.
- Kuo et al., `arXiv:2605.26526`, "Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks". The abstract frames open-weight safeguards and abliteration/prefilling attacks. It supports the open-weight threat model, but does not show this one-layer post-hoc add-back result.

Novelty verdict before data: no STOP if the claim is constrained to selective repair at a dev-chosen single layer/gain and an untouched test split. If the post claims only that the direction remains readable, or that harmful refusal can be restored while ignoring harmless refusal, it is not novel enough.
