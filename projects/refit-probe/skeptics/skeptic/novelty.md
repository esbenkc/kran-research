# Novelty check (Vera, skeptic) — refit-probe

Checked 2026-10-19 (CEST night) against the abstracts and the claims the room quotes. Own reads of the arXiv abstracts.

| Source | What it already shows | What it does NOT show |
|---|---|---|
| Wehner & Fritz, arXiv 2510.21531 | Probe-based DPO for toxicity; "simply retraining probes after optimization recovers high detection accuracy" (0.957-0.973 per their text). | Deception; a label-count curve; a held-text control (base weights reading the policy's answers). |
| Libon et al., arXiv 2609.38645 | Training against frozen probes is "easily exploitable"; "concepts stay linearly encoded after fine-tuning"; from-scratch refits for honesty near base (Table 2, max over layers, continuously updated regime, per Iris/Kasper quote check #2921). | Refit after training against a FROZEN probe with a reward-only DPO penalty; label-count curve; held-text control. |
| Taufeeque et al. (Obfuscation Atlas), arXiv 2602.15515 | Taxonomy obfuscated activations vs obfuscated policy; App. L: continual retraining during RL makes no significant difference (off-domain refit data). | On-policy refit with n labels after training. |
| Cundy & Gleave, arXiv 2505.13787 | A lower decision boundary on the frozen lie detector still detects an evading policy, at the cost of more false positives. | Refit; label count. |
| Bailey et al., arXiv 2412.09565 | Obfuscated activations are abundant; adversarially retrained monitors are bypassed again by a new attack. | Post-hoc refit on a DPO-trained policy without a new attack. |
| Our own /probe-ensemble (kran-research oracle-1018) | Off-policy refit (600 DolusChat replies, attention probe L16) at round 8: 0.72-0.82 vs base 0.79 (data/refit.json; I re-read the file). | On-policy labels; n-curve; refit at the trained-against layer 12. |
| Our own /rethreshold-probe-evasion (theory-1018) | New threshold on 200 clean examples wins back most of the lost catch rate under a different attack (Qwen2.5-1.5B). | Refit vs re-threshold at equal n on DPO-trained policies. |

New in this run (if the data hold): (1) refit AUROC as a function of the number of on-policy labels (25-400) on DPO
policies trained against a frozen deception probe, (2) the held-text arm separating "the model changed" from "the
answers changed", (3) refit vs re-thresholded old probe at equal label budget. The bare claim "a refit recovers
detection" is NOT new (Wehner & Fritz; Libon et al.).
