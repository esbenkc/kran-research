# Independent Novelty & Literature Check — skeptic3 (Nadia)
Room: oracle-1019-refit-probe
Date: 2026-10-18 / 2026-10-19

## 1. Literature Survey & Grounding

I independently fetched and audited the published literature regarding probe retraining and monitor evasion:

### A. Wehner & Fritz (arXiv:2510.21531, Oct 2025)
*Probe-based Fine-tuning for Reducing Toxicity*
- **Claims**: Evaluates SFT and DPO against linear probes for toxicity reduction. Finds "simply retraining probes after optimization recovers high detection accuracy" (0.977–0.987). Concludes probe ensembles are largely unnecessary if retraining is feasible.
- **Boundaries**:
  - Focuses on toxicity (style/attribute), not factual deception or intentional lying.
  - Does not evaluate sample complexity of the refit (i.e. accuracy as a function of label budget $n$).
  - Does not evaluate whether retraining provides any benefit over simple re-thresholding of the frozen probe.

### B. Libon et al. (arXiv:2609.38645, Sep 2026)
*Alignment via Training Against Probes Without Losing Monitorability*
- **Claims**: Evaluates probe-guided fine-tuning for harmlessness and honesty. In Table 2 and Appendix B.3–B.5, reports that under continuously updated probe training, honesty concepts remain linearly encoded (probe AUROC 0.99 -> 0.98 on Qwen-14B). Observes that evasion of static probes often manifests as a common-mode downward translation of activation scores for both classes while preserving linear separability.
- **Boundaries**:
  - Focuses primarily on training models *with* continuously updated probes to maintain alignment, rather than auditing post-hoc frozen DPO evasion.
  - Does not provide label sample-efficiency curves ($n \in [25, 400]$).
  - Evaluates best-layer selection rather than evaluating the exact layer subjected to the training penalty.

### C. Taufeeque et al. (arXiv:2602.15515, Feb 2026)
*The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes*
- **Claims**: Formalizes the distinction between *obfuscated activations* (internal representation drift) and *obfuscated policy* (outputting deceptive text structured to evade the detector). Finds that probe penalties incentivize obfuscated policies rather than internal activation hiding. In Appendix L, shows that retraining detectors on off-domain data does not overcome evasion.
- **Boundaries**:
  - Investigates RLVR (coding/reward hacking) rather than conversational chat / DolusChat.
  - Appendix L retraining used off-distribution data; does not evaluate in-distribution label efficiency.

### D. Cundy & Gleave (arXiv:2505.13787, May 2025)
*Preference Learning with Lie Detectors can Induce Honesty or Evasion*
- **Claims**: Introduces DolusChat. Demonstrates that GRPO can yield policies with >85% evasion against frozen deception probes. Section 5.5 explicitly notes that evading responses could still be detected by the frozen detector using a lower decision boundary (re-thresholding) at the expense of higher false positives.
- **Boundaries**:
  - Did not test probe refitting or retrained direction vectors; only noted ROC curve trade-offs on the original probe.

### E. Prior Work from Our Own Lab
- `probe-ensemble` (2026-10-18): Noted off-policy refit of attention probe yielded AUROC 0.72–0.82 vs base 0.79, and alarms dropped from 84% to 20–27% while AUROC remained 0.72–0.76.
- `rethreshold-probe-evasion` (2026-10-18): Demonstrated that adjusting the threshold on 200 clean examples recovered 90% of lost catch rate, as the score distribution underwent a common-mode shift.

---

## 2. Assessment of Novelty

1. **What is NOT Novel (Must NOT be claimed as unprecedented)**:
   - The bare qualitative claim that "a retrained probe can still detect deception/misbehavior after optimization" is **not novel**. It was observed by Wehner & Fritz (2025) for toxicity, Libon et al. (2026) for dishonesty, and previewed in our own `probe-ensemble` lab notes.
   - The claim that frozen probes retain ranking and can be partially restored by shifting thresholds is **not novel** (Cundy & Gleave §5.5, Libon App. B.5, and `rethreshold-probe-evasion`).

2. **What IS Truly Novel and Groundbreaking**:
   - **Label Sample Efficiency ($n$-scaling)**: Measuring the exact label scaling curve ($n \in \{25, 50, 100, 200, 400\}$) required to refit an auditor on an evading policy.
   - **Direct Decomposition of Mechanism**:
     - (a) Old probe @ old threshold (the evasion baseline)
     - (b) Old probe @ re-thresholded boundary (isolating score translation)
     - (c) On-policy refit @ budget $n$ (measuring directional rotation vs translation)
     - (d) Held-text control (untrained base model reading policy text, isolating text drift from internal representation obfuscation)
   - **Resolving the Mechanism**: Resolving whether DPO evasion in DolusChat represents true latent obfuscation (Bailey et al.) or an obfuscated policy altering output vocabulary (Taufeeque et al.).

## 3. Reviewer Verification Standards for Signoff

To sign off on CONFIRMED or REJECTED:
- Verification must strictly execute on saved raw test outputs and activations.
- Results must NOT conflate frozen re-thresholding recovery with refit recovery.
- Verification must confirm whether refit AUROC beats text-only (bag-of-words) and shuffled-label nulls.
