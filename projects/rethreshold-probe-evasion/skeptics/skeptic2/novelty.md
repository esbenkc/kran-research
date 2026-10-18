# Novelty and Literature Context - Hugo (skeptic2)

## 1. Prior Literature and Precedents

### Bailey et al. (2024), "Obfuscated Activations Bypass LLM Latent-Space Defenses" (arXiv:2412.09565)
- **Prior finding:** Latent-space monitors (linear probes, SAEs) are vulnerable to optimization against the monitor loss, reducing recall to 0% at low false-positive rates while maintaining output behavior via retain penalties.
- **Connection to thresholding:** In Appendix F ("Why recall at low FPR?"), Bailey et al. explicitly assume that a defender re-thresholds on benign data: *"picking a threshold that achieves a 1% FPR is something a realistic defender can do, since the defender has access to benign (i.e., negative) samples ... They can simply compute monitor scores on this benign data and pick a decision threshold such that around 1% of those scores are above the threshold."*
- **Distinction:** Bailey et al. use this benign-calibration procedure to define the metric (recall at 1% FPR), but they do not decompose the apparent evasion into a rigid translation of representation space versus true geometric destruction of class separation, nor do they quantify the proportion of TPR recovered by recalibration across different attack strengths.

### Libon et al. (2026), "Alignment via Training Against Probes Without Losing Monitorability" (arXiv:2609.38645)
- **Prior finding (App. B.5):** They observe that training against a frozen linear probe evades detection primarily through a "translation of the representation across its fixed boundary." They demonstrate that retraining a fresh probe on the attacked activations restores high AUROC.
- **Distinction:** Libon et al. do not report the performance of the *frozen* probe when only its scalar threshold is adjusted; they evaluate fresh probe re-fitting. Our experiment separates scalar threshold shift ($c$) from representation rotation or collapse.

### Bosco & Srinivasan (2026), "Locating and Steering Refusal Beyond Attention" (arXiv:2609.04721)
- **Prior finding (App. A.28):** They report that adversarial suffixes evading a mean-pooled probe at its default sigmoid midpoint threshold are caught (1.00 TPR) once the threshold is re-calibrated on harmless prompts at 1% FPR.
- **Distinction:** Their finding demonstrates threshold shift for token-level discrete suffix attacks. Our study tests whether weight-space gradient optimization (LoRA fine-tuning) exhibits the same shift dominance or whether it erases the underlying geometric separation.

### Tong, Feng, & Li (2018), "Neyman-Pearson classification algorithms" (Science Advances / arXiv:1608.03109)
- **Prior statistical theory:** In the Neyman-Pearson umbrella framework, setting a threshold using the order statistics of $n$ held-out negative samples to guarantee $\text{FPR} \le \alpha$ with confidence $1 - \delta$ requires sample size:
  $$n \ge \left\lceil \frac{\log \delta}{\log(1 - \alpha)} \right\rceil.$$
  For $\alpha = 0.01$ (1% FPR) and $\delta = 0.05$ (95% confidence), this bound requires $n \ge 299$ clean negatives.
- **Application here:** With $n=200$ clean negatives (2nd-largest rule), the empirical threshold has an expected out-of-sample FPR of $\frac{2}{201} \approx 0.995\%$, but carries non-trivial finite-sample estimation variance ($P(\text{realized FPR} > 1\%) \approx 0.41$). This statistical sampling spread explains why empirical recovery can undershoot the theoretical binormal prediction even when internal class ranking is preserved.

---

## 2. Scope and Claims of This Work

- **Measured phenomenon:** We quantify the fraction of lost true positive rate at 1% FPR recovered by simple threshold recalibration on clean negative data after a LoRA fine-tune against a linear probe.
- **Finding:** When an output retain term is applied ($\lambda \ge 10^4$), evasion is overwhelmingly dominated by a common-mode translation ($c \approx 0.90\text{--}0.96$), leaving AUROC virtually unchanged ($\Delta\text{AUROC} < 0.001$). Under these conditions, recalibration recovers $\approx 85\text{--}94\%$ of the lost TPR.
- **Boundary condition:** When no retain penalty is enforced ($\lambda = 0$), the attacker drives the separation distance to near zero, causing recalibration recovery to collapse to $\approx 1\%$.
