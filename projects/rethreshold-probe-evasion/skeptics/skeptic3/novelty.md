# Novelty Check: Probe Evasion and Threshold vs Separation

**Claim:** We measure how much of the lost TPR at 1% FPR a threshold re-fit alone gets back for an unconditional weight-space probe-evasion fine-tune, decomposing the evasion into a coordinate shift vs. separation loss.

**Verification:**
1. **Bailey et al. (2024, arXiv:2412.09565):** Examines obfuscated backdoors. Appendix F notes that "picking a threshold that achieves a 1% FPR is something a realistic defender can do, since the defender has access to benign (i.e., negative) samples," assuming defenders take this step to justify their metric. However, they do not split the evasion drop of a *frozen* probe into these two components.
2. **RL-Obfuscation (2025, arXiv:2506.14261):** Uses reinforcement learning to evade monitors. The metric used is a drop in recall or AUROC, without decomposition.
3. **Locating and Steering Refusal Beyond Attention (Bosco & Srinivasan 2026, arXiv:2609.04721):** Appendix A.28 states: "Even the mean pool catches the suffix once its threshold is benign-calibrated rather than set at the sigmoid midpoint... it fires on 1.00 of evasion prompts at the same false-positive rate." This highlights threshold sensitivity but applies to a suffix attack, not weight-space.
4. **Alignment via Training Against Probes (Libon et al., 2026, arXiv:2609.38645):** Appendix B.5 shows that a frozen linear probe is evaded by translating representations across the boundary. They identify the translation mechanism but do not report the frozen probe with a new threshold.

**Conclusion:** Prior studies either assume the defender re-thresholds or identify the shift mechanism qualitatively. We quantify the explicit split of lost TPR into a threshold shift vs. separation loss to identify how much of the evasion effect is simply a coordinate shift.
