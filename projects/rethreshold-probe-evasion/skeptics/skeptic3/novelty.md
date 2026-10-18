# Novelty Check: Probe Evasion and Threshold vs Separation

**Claim:** No paper has quantified the split of lost TPR (at a fixed FPR) into threshold shift versus separation loss for an *unconditional weight-space* probe-evasion fine-tune.

**Verification:**
1. **Bailey et al. (2024, arXiv:2412.09565):** Examines obfuscated backdoors. They note that evasion persists even if the defender trains a new monitor. They do not decompose the evasion of a *frozen* probe into a coordinate shift vs. separation loss.
2. **RL-Obfuscation (2025, arXiv:2506.14261):** Uses reinforcement learning to evade monitors. The metric used is a drop in recall or AUROC, without further decomposition into threshold-shift and separation-loss components.
3. **Locating and Steering Refusal Beyond Attention (Bosco & Srinivasan 2026, arXiv:2609.04721):** Appendix A.27 shows that a dilution attack in the input space leads to a pure threshold shift (recalibrated mean pool fires correctly). Appendix A.28 shows a suffix attack evades a mean-pooled probe, but recalibration here was on a different threshold (sigmoid midpoint), not a strict 1% FPR. This relates to input-space, not an unconditional weight-space fine-tune.
4. **Alignment via Training Against Probes (Libon et al., 2026, arXiv:2609.38645):** Appendix B.5 shows that a frozen linear probe is evaded by translating representations across the boundary, while a fresh probe recovers AUROC. They identify the mechanism (translation) but do not explicitly quantify the *recovered TPR share at 1% FPR* using a clean-negative threshold re-fit.

**Conclusion:** The proposed post provides the first quantified split of lost TPR into threshold shift vs. separation loss for an unconditional weight-space attacker, directly answering how much of the evasion is just a coordinate shift. Novelty confirmed.