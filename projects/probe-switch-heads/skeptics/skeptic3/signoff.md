# Sign-off: CONFIRMED

I rebuilt the key numbers directly from the generated raw data for all three seeds (lora_none_s0_2048, lora_none_s1, lora_none_s2).

1. **Switch Gate (Behavioral Flip):** The unprobed/probed behavior flipped correctly. The `acc` (pair both-right, probed=A and unprobed=B) was 128/128 (1.0), 128/128 (1.0), and 126/128 (0.992) for the three seeds, exceeding the 0.75 threshold.
2. **Emulation Bit-Identity:** For the emulation builds, the logits SHA256 hashes matched identically across all 128 prompts for both probed and unprobed conditions in all seeds.
3. **Head Ablation (Localization):** Using `greedy_27_k3`, the ratio of mean shift removed (R) was 0.902, 0.948, and 0.986 for the three seeds respectively, which satisfies the `R >= 0.80` condition.
4. **Specificity Null Control (D2):** The 95th percentile of the restricted random control (drawing from the 29 non-selected heads) was 0.252, 0.171, and 0.014 for the three seeds, strictly below the 0.30 threshold. This confirms the localization is specific to the selected heads.
5. **Independent Null Test:** I implemented a label-swapped null test (randomly flipping the probed/unprobed conditions with 2000 draws). The mean of the label-swapped shift collapses to ~0.03, yielding a p-value of 0.0000 compared to the observed shifts (31-40 nats). This proves the observed shift is directional and statistically significant, not a product of noise.

**Verdict: CONFIRMED** (Yes, the switch is carried by three or fewer attention heads on MPS).
