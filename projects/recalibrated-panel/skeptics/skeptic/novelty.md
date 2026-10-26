# Novelty check (Vera, skeptic, 2026-10-26)

Claim under test: a LoRA fine-tune of Qwen2.5-1.5B-Instruct, trained against a panel of 20 frozen clean-fit
country-token probes in which each member is re-standardised INSIDE the loss on fresh true statements of the
current model (gradient through the spread), at a pre-fixed output-matching KL weight; scored by (a) the same
re-standardised max rule with a 1% line on held-out true statements of the attacked model, and (b) labelled
ceiling heads (logistic, QDA) on the 20 frozen scores, at the 1% line.

What I read myself (abstracts or full text, 2026-10-26):

- Scholten and Günnemann, "Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning",
  arXiv 2410.09878 (ICLR 2025). Poisoning of TRAINING and CALIBRATION DATA for conformal sets; defence =
  calibrate on disjoint subsets and take a majority. Here the attacker owns the SCORER (the model's weights), not
  the calibration data, so their disjoint-subset defence does not apply: every subset is scored by the attacked
  model. The analogy is credit, not prior result.
- Chaudhary and Barez, "Beyond Black-Box Obfuscation", arXiv 2505.14300. Red-team evasion of white-box monitors;
  finds information migrates between linear and non-linear subspaces and covariance manipulation; an 'any'
  ensemble (SafetyNet) of linear, non-linear, PCA and Mahalanobis detectors. Not a recalibrated fixed probe panel;
  not a weight-space attacker with a fixed output-matching price.
- Aremu, Ognev, Poppi, Lukas, "Adaptively Robust LLM Monitoring via Activation Watermarking", arXiv 2603.23171.
  Adaptive (prompt-search) attackers evade every baseline monitor >= 79% at matched 1% FPR. Input-space attacker;
  no recalibration of the monitor on the attacked model.
- Bailey et al., arXiv 2412.09565 (obfuscated activations; mean-over-layers aggregation attacked by soft prompts),
  Westphal et al., arXiv 2606.09411 (periodically REFIT probes, all-layer rule), Gupta and Jenner, arXiv 2506.14261
  (RL-Obfuscation, max over tokens): read for theory-1022 (my novelty.md there); none attacks a per-member
  re-standardised panel with the moments in the loss.
- Samuelson 1968 (JASA 63:1522) and Chamness and Traylor, arXiv 1601.07521: the k-th order statistic bound
  sqrt((n-1)(n-k)/(nk)) for the sample sd; with ddof 0, sqrt((n-k)/k). Known result; the post uses it, it is
  not new. Re-derived and checked: vera_lemma.py / vera_lemma.out.

Search (web, 2026-10-26): "adaptive attacker fine-tuning against recalibrated linear probe ensemble",
"fine-tuning evade probes recalibrate threshold deployed model": no paper found that trains a weight-space evader
against a re-standardised (per-member z-scored on the model's own true statements) multi-layer probe panel, or that
measures labelled 20-score ceiling heads on the attacked model. Our own earlier Lab posts are the nearest:
rethreshold-probe-evasion (re-thresholding after a one-probe fine-tune), panel-fine-tune (theory-1022; the post-hoc
Rule B restore 60-69% that this run attacks).

Verdict: new as far as I can find. Credit 2410.09878 for the poisoning frame, 2505.14300 and 2603.23171 for
ensemble / adaptive-attacker evaluations at a matched 1% FPR, Samuelson / Chamness-Traylor for the bound.
