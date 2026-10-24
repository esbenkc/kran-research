# Novelty Check (skeptic3)

I confirm that testing a trained output-matched evader against a probe explicitly restricted to output-sensitive directions (via Fisher Information projection) is novel. 

- FishBack (Wang et al., arXiv 2605.17231) provides the theoretical floor for the minimum-norm output movement, but does not test it as a defense mechanism via projection.
- Park, Nief, Choe, Veitch (arXiv 2602.15293) identify the rank deficiency of the Fisher matrix in the final layer.
- Zhong (2026, LessWrong) tests spurious probes acting as a black-box alternative, but without the explicit output-matched fine-tuning loop against the cut probe.
- Other adversarial evasion papers (e.g., RL-Obfuscation, Neural Chameleons) keep the probe geometry fixed and do not restrict to output-curvature.

The experiment and its finding—that the shared-weight fine-tune pays the same output change regardless of the probe's theoretical minimum cost—is novel.
