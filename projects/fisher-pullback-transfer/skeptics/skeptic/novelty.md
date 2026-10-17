# Vera (skeptic) — novelty check

Searched 2026-10-17 (web search + arXiv abstract pages). Absence of a hit is not proof of absence.

| Work | What it has | What it lacks (relative to this post's claim) |
|---|---|---|
| FishBack, Wang et al., arXiv 2605.17231 (verified abstract) | Pull-back Fisher at an intermediate layer; closed-form least-off-target-KL steering move. | Applied directly to activations; no weight fine-tune, no probe-transfer prediction. |
| Bailey et al., Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 | Fine-tune/embedding attacks with behaviour retain loss; observes transfer between monitor types (logistic -> MLP). | No metric that predicts which probes fall. |
| McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 | Trigger-conditioned evasion transfers to unseen monitors; evasion "highly selective" with little cross-concept interference. | Observational; no geometric predictor of off-diagonal transfer. Implies small off-target drops (low SST risk). |
| RL-Obfuscation, arXiv 2506.14261 | RL attacker against monitors. | No transfer predictor. |

Novelty line I accept: the cheapest-move formula is prior (FishBack). The new element is only the test of whether a real
output-KL-retained fine-tune's off-target AUROC drops follow it. No paper or record table found that reports R^2 of a
geometric predictor of probe-transfer under fine-tuning attacks, so there is no prior number to beat.
