# Novelty check — skeptic3

Question: under a probe penalty, does the *type* of deceptive answer shift from falsification to
omission / misleading-but-true, so that judged lying falls while deception stays?

Searched 2026-10-16 (arXiv full text grepped for omission / lie type / misleading / evasion).

## Closest prior work (all public arXiv)

1. **Hollinsworth, Dombrowski, Adam-Day, Gleave, Cundy (2026), "Scaling Trends for Lie Detector
   Oversight in Preference Learning", arXiv 2607.01567.** Closest. In an appendix they compare a
   binary judge (v0) to a stricter "ternary" judge (v1) on DolusChat-style preference learning with
   a lie detector (SOLiD). For the *policy* trained under detector pressure (TPR 0.7), v0 calls
   57.5% deceptive and v1 74.0%; on the 33 disagreements a human sides with v1 28/33. They write the
   policy "often uses partial omissions that v0's binary framing misclassifies as truthful" and
   "lies of omission ... are particularly salient for the policy which has been trained with
   pressure to deceive". So the *qualitative* claim "detector-trained policies lean on omission,
   and a falsification-only judge undercounts it" is already in print.
   Differences from this run: preference learning (detector picks labels), not a direct probe
   penalty in the reward; no per-round lie-type composition; no lambda=0 control isolating the
   penalty from training itself; 8B+ models, not ~1B.
2. **Cundy & Gleave (2025), "Preference Learning with Lie Detectors can Induce Honesty or
   Evasion", arXiv 2505.13787 (NeurIPS 2025).** Introduces DolusChat with `lie_type`
   (omission / fabrication / exaggeration / minimization). Qualitative examples: GRPO policies
   evade with "evasive language or partial truths" that get lower lie scores. No quantified
   lie-type shift over training.
3. **Taufeeque, Heimersheim, Gleave, Cundy (2026), "The Obfuscation Atlas", arXiv 2602.15515.**
   Taxonomy: honest vs obfuscated activations vs *obfuscated policy* (deceptive text that evades
   the detector, usually by adding a justification). A lie-type shift is one specific kind of
   obfuscated policy. Coding / reward-hacking setting, no omission analysis.
4. **Thormann (2026), "Probing the Limits of the Lie Detector Approach to LLM Deception", arXiv
   2603.10003.** Truth probes detect lies much better than misleading non-falsities. Static
   (no training against the probe). Supplies the mechanism that would make a shift pay off.
5. **Moustafa, Feser, Mai (2026), "Beyond Liars' Bench", arXiv 2607.20479.** Lie typology
   (fabrication / omission / exaggeration) changes probe detectability. Static, no training.
6. Context only: "Deception by Omission: Language Models Knowingly Hide Their Mistakes", arXiv
   2610.11351 — omission as a natural LLM behaviour, not under probe pressure.

## Verdict

**Partly novel.** The idea that detector pressure pushes policies toward omission is not new
(Hollinsworth et al. 2026 report it as a judge-calibration observation; Cundy & Gleave 2025 show
examples). What this run adds, if it holds up: (a) the direct probe-penalty setting
(probe-goodhart), (b) a per-round lie-type composition, (c) a lambda=0 control that separates the
penalty's effect from training drift. A null (no shift) is also new information, because the
prior evidence is anecdotal / appendix-level.

Required for the post: cite Hollinsworth et al. 2026 as the closest prior observation and say
plainly what is new. Do not claim to be the first to notice omission under lie-detector
pressure.
