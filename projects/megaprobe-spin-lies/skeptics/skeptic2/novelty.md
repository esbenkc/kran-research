# Novelty and Definition Audit (Hugo, skeptic2)
Room: oracle-1020-megaprobe-spin-lies
Date: 2026-10-20

## 1. Prior Art and Grounding

The experiment addresses whether a multi-layer megaprobe recovers deception detection performance on "spin lies" produced by models trained against a lie detector.

### Prior Work Verified:
1. **Nordby et al. (arXiv 2604.13386)**:
   - "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling".
   - Shows that a 5-layer linear ensemble achieves a +12.7% relative gain (0.827 -> 0.932 AUROC, absolute +0.105) on Llama-70B across 5 tasks.
   - Mechanism: stacking linear probes with logistic regression across layers 20-40.
   - Crucial qualification: 2 of 5 tasks performed worse (-5.3% on Instructed Deception, -4.2% on Harm-Pressure Choice).
   - Reports gradual rotation of deception directions across layers, but tested linear combinations only.

2. **Taufeeque, Heimersheim, Gleave, Cundy (arXiv 2602.15515, "The Obfuscation Atlas")**:
   - Proposes follow-up question readout: appending `User: Did you respond truthfully? Assistant: Yes` and reading the final token residual stream.
   - Per-layer logistic regression probe with a cross-entropy training loss filter (< 0.6), combined via mean of sigmoids.
   - Found that reinforcement learning against a detector penalty primarily incentivizes "obfuscated policies" (deceptive text that evades the detector, such as providing justifications) rather than obfuscating internal representations.

3. **Hollinsworth, Spies, Diriba, Gleave, Cundy (arXiv 2610.12445, "Telling FIBS")**:
   - Introduces the public multi-layer and token probe architectures in github.com/AlignmentResearch/fibs-probes.
   - `LinearEnsemble`: 6-layer linear probe averaging sigmoids with loss threshold < 0.6.
   - `AxialProbe`: cross-layer and cross-token attention probe with shared projection, rotary token attention, and layer-axis attention.
   - Trained on 152,980 rows with early stopping on 5,000 validation rows.
   - Crucial distinction: FIBS evaluated probes on models without training the models against lie detectors.

4. **Kramár et al. (arXiv 2601.11516)**:
   - Attention-pooled probes over tokens using softmax attention weights.

5. **Gupta & Jenner (arXiv 2506.14261)**:
   - Showed that RL fine-tuning easily evades mean/median-pooled probes, whereas max-pool and attention-based probes exhibited greater empirical robustness on 7B-14B models.

## 2. What Is Truly Novel vs Prior Art

### NOT Novel:
- Stacking linear probes across layers (Nordby et al. 2026; Taufeeque et al. 2026).
- Follow-up question readout at the final token (Goldowsky-Dill et al. 2025; Taufeeque et al. 2026).
- Attention-pooling over tokens (Kramár et al. 2026).
- Cross-layer and cross-token axial attention probes (Hollinsworth et al. 2026, FIBS AxialProbe).
- That RL against lie detectors drives models toward textual evasion/obfuscated policies (Taufeeque et al. 2026).

### Genuine Novelty of this Study:
- **Sample-efficient refitting under adversarial training**: Testing whether an attention megaprobe (or FIBS axial probe) refit on a realistic budget of 400 labels can recover detection of evasive responses on models explicitly trained against a lie detector (DPO policies).
- **Subclass evaluation on fact-word-retaining lies ("spin lies")**: Evaluating whether higher probe capacity recovers detection on deceptive responses where the model retains awkward fact words, vs whether the failure is inherent to representation collapse or text-level evasion.
- **Direct controlled comparison**: Evaluating linear refit, Atlas layer-averaging, attention megaprobe, and FIBS axial probe under identical 5-fold GroupKFold splits on the same dataset.

## 3. Definition Checks and Scope Boundaries

### Check 1: Definition of "Spin Lies"
- **Claimed Narrative**: "The answer states the awkward fact and then frames it to mislead."
- **Operational Reality**: The class is defined by `y_judge == 1` (judge calls response DECEPTIVE) and `fact_score >= 0.27` (answer contains >= 27% of key words from the true fact).
- **Audit via Agent Labels (Blind Claude audit of 50 judge-spin samples)**:
  - 14/50 (28%): Framing-only spin (states the fact and spins it).
  - 28/50 (56%): Fact-word contradiction (mentions fact keywords but contradicts, negates, or confuses the fact).
  - 8/50 (16%): Appears honest / judge false-positive.
- **Verdict on Definition**: The colloquial label "spin lies" over-describes the class. The text must explicitly define the class as "deceptive answers containing at least about a quarter of the awkward fact's key words" and disclose the 14/28/8 empirical breakdown.

### Check 2: Definition of Negatives
- In /refit-probe, the 0.60-0.69 AUROC baseline was measured against **all truthful answers**.
- In this study, the verdict arm specifically tests against **fact-stated truthful answers** (`y_judge == 0` and `fact_score >= 0.27`).
- On the base model, linear probe AUROC is 0.722 against fact-stated truthful, but 0.642 against all truthful answers. The gap (~0.08) demonstrates that linear probes partly rely on "fact absent" cues. Reporting must not conflate these two negative sets.

### Check 3: Positive Control Invariant
- The pre-registered protocol (PREREG.md s6) requires that the candidate megaprobe match or beat the linear probe on all lies on the base model (positive control).
- Base model evaluation shows `c_all` (megaprobe) = 0.810 vs `a_all` (linear refit) = 0.852.
- The positive control fails on probe (c). A capacity increase on 400 labels leads to sample-inefficiency/overfitting on 896 dimensions x 24 layers, underperforming simple linear ridge/logistic regression.
