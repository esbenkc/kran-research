# Novelty check — Vera (skeptic), 2026-10-18

Claim to check: "Nobody has split a weight-space (fine-tune) probe-evasion drop into a threshold shift
(undone by re-fitting the threshold on clean negatives of the attacked model) and a loss of separation."

## Sources checked (all opened, text searched)

| Source | What it does on thresholds | Overlap with our question |
|---|---|---|
| Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*, arXiv 2412.09565 (html) | Recall at 1% FPR; threshold picked on benign samples; bootstrap CIs over the threshold set (Sec 2.2, App. "why 1% FPR"). Sec 4.2 obfuscated backdoor: monitors get one incremental update per attacker step on clean data, and after training "the defender trains a new monitor from scratch for the final model, and we evaluate this new monitor". Sec 4.3 data poisoning: OOD detectors are fit on the backdoored model's benign activations. | PARTIAL. For their weight-space attacks the defender already re-fits (whole monitor, not only the threshold) on the attacked model, and evasion survives. That is a stronger defence than ours and a stronger attacker (it trains against the re-fitting monitor). They never report the frozen-probe drop split into shift vs separation. The post must NOT say "weight-space evasion was never tested against recalibration"; it may say "nobody split a frozen-probe fine-tune's drop into a threshold part and a separation part". |
| Gupta & Jenner, *RL-Obfuscation*, arXiv 2506.14261 (html) | Recall at 1% FPR drop for probes on an RL-evading model; no grep hit for recalibrat* or for thresholds re-fit on the evading model's benign data. | No decomposition. |
| arXiv 2609.04721 (*Locating and Steering Refusal Beyond Attention*), App. A.27/A.28 (Tove's check, #2648) | Input-space only: dilution attack is a threshold failure (recalibrated mean pool recovers); crafted suffix needs max/harm-weighted pooling. | Input-space analogue; no weight-space attacker. |
| arXiv 2608.16852 (Sadhu et al., compliance-detector audit) | Rule blindness; ICS readout; per-domain re-calibration. | Does NOT contain the 6.7%->95% suffix result cited in the goal. Do not cite it for that. |
| arXiv 2609.38645, *Alignment via Training Against Probes Without Losing Monitorability* (html, 2 weeks old) | Frozen probes are evaded; "the target property remains linearly decodable after training: a freshly trained probe reaches AUROC comparable to one trained on the base model". No grep hit for threshold recalibration / FPR re-fit. | CLOSEST. Covers Tove's label-refit probe segment (fresh probe, labels) for a weight-space attacker. It does not test a negatives-only threshold re-fit of the frozen probe. The post must cite it next to the probe-refit bar, and that bar is not new. |

## Verdict
The negatives-only threshold re-fit of a FROZEN probe after a weight-space fine-tune, with the drop split into
a shift part and a separation part, is new as far as these five sources and one web search show. Two framing
limits: (1) cite Bailey Sec 4.2 (defender retrains from scratch, evasion survives) as the strong-attacker
counterpoint; (2) the label-refit probe result is already in 2609.38645, so it is a replication, not a finding.

## Update after the room (Iris #2703/#2737, checked against my own reads)
- Libon et al., arXiv 2609.38645 App. B.5: LoRA + KL against a FROZEN linear probe; both class means fall, "the frozen
  probe is thus evaded by a translation of the representation across its fixed boundary". That is the qualitative version
  of our λ = 1e5 result. Not new: the translation. New here: the measured share of lost TPR at 1% FPR that a negatives-only
  re-fit recovers (0.90 → 0.76 → 0.61 → 0.01 as the push grows), on fixed inputs, with the λ = 0 collapse.
- Bailey et al. App. F already assume a defender who picks the 1%-FPR threshold on benign data. So the post must not say that
  Bailey et al. score evasion at a stale threshold.
- Final check (my rebuild): the headline is not luck in our favour; the frozen refit-200 share (0.904) is below the 2.5% quantile (0.922) of 2000 redraws.
