# Novelty check (Vera, skeptic) — oracle-1021-refusal

Claim under test (Tove's reshaped title): adding the BASE model's refusal direction back at ONE layer
of a refusal-stripped Qwen2.5-1.5B-Instruct restores >= 80% of the base harmful refusal rate while
harmless refusal stays < 5% (layer + gain frozen on dev, scored once on test).

Checked from full text in swarm-oracle-1021-refusal/lit/ (grep offsets given).

| Paper | What it shows | Overlap with our claim |
|---|---|---|
| Lee et al. 2610.00320 (2026-09-29) | Sec 4.1: frozen clean-model probe still separates harmful/benign after a harmful LoRA (AUROC 0.75-0.997, 6 checkpoints). App H (char ~84140): "adding the clean model's refusal direction into the residual stream over contiguous bands, four gains and two write rules": Llama-3.1-8B best 0.37 vs clean 0.92; one Qwen2.5-14B arm 0.98 vs 0.99 but refuses 0.66 of benign prompts (clean 0.00). Also single-layer full-state transplant and its rank-1 summary (top singular vector of clean-minus-compromised states, NOT the base diff-of-means direction). | Closest prior. Readability (AUROC >= 0.9) is ALREADY shown, so it is not a finding here. Write-back was done over BANDS, no per-layer profile, no benign-budgeted operating point, 8-14B only, one appendix paragraph. |
| Arditi et al. 2406.11717 | Refusal mediated by one diff-of-means direction; ablation removes refusal, ADDING it to harmless prompts induces refusal in the aligned model. | Shows add-back over-refuses on the BASE model; never on a fine-tune-stripped model. |
| Glass et al. 2608.24988 | SFT reverses steered behaviour via updates near-orthogonal to the steering direction (rho 0.004, cos 0.074). | Different direction of the question (does FT undo a steer); never re-adds the base direction to a stripped model. |
| 2604.27019 (Dynamic Adversarial FT Reorganizes Refusal Geometry, Mistral-7B) | A KL-constrained "carrier" restores refusal at late R2D2 checkpoints but "increases over-refusal and damages benign utility". | Adversarial-training checkpoints, not a harmful-compliance strip; carrier is searched, not the base diff-of-means direction. Second prior data point that add-back costs selectivity. |
| Fool's Gold 2608.17202 | Defensive deception vs safety-removal; "benign fine-tuning never restores refusal". | No activation add-back. |
| 2605.26526 (FT defenses susceptible to simple attacks) | Defence survey/attacks; cites CAA only. | No add-back result. |
| 2507.11878 (harmfulness and refusal encoded separately) | Diff-of-means at the instruction tokens = harmfulness; at post-instruction tokens = refusal; they differ. | DEFINITION RISK for us: our "refusal direction" must be taken at the post-instruction (chat-template) position like Arditi, else we add a harmfulness direction. Post must state the token position. |

Verdict: NO STOP. The single-layer, per-layer, benign-budgeted (< 5% harmless, dev-frozen) add-back of the
base diff-of-means direction into a harmful-LoRA-stripped 1.5B model is not in these papers. The post must
credit Lee et al. App H as the one prior write-back data point (band-wise, over-refused 0.66 on Qwen2.5-14B)
and must NOT present readability as new.
