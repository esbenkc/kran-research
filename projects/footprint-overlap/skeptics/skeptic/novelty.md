# Novelty check (Vera, skeptic), 2026-10-27 PDT

Claim to check: a score computed from the bits of a monitor build, before any behaviour test, predicts the share of a
watched-only model's behaviour change that a NEW monitor build triggers (the "fooled share").

Searched (web_search, 2 queries; read abstracts):
- Bailey et al., "Obfuscated Activations Bypass LLM Latent-Space Defenses", arXiv:2412.09565 (ICLR 2026):
  obfuscation often transfers between monitor types (logistic -> MLP probes), not from probes to OOD detectors.
  Reports transfer per pair; gives no predictor of which monitors transfer. Their obfuscation hides a representation;
  it does not sense the monitor's presence.
- Chaudhary & Barez, "Beyond Black-Box Obfuscation", arXiv:2505.14300: mechanisms of evasion (geometric shifting,
  covariance manipulation); an ensemble defence. No pre-behaviour transfer predictor.
- Das et al., "You Can't Escape Your Own Activations", arXiv:2609.03035: telling agents about a monitor does not hide
  them from probes. No trained monitor-sensing model; no transfer law.
- Mosafer, "From Latent Space to Jacobian Space", arXiv:2610.04316: monitor-aware suffixes suppress a prompt-side
  monitor; no cross-monitor transfer predictor.
- Neural Chameleons arXiv:2512.11949 and RL-Obfuscation arXiv:2506.14261 (from Iris's research.md, not re-read by
  me beyond the search snippets): zero-shot evasion of unseen monitors; no bits-level predictor.
- Demontis et al. 2019 (USENIX Sec, arXiv:1809.02861): gradient alignment predicts adversarial-example transfer
  between classifiers. Closest analogue in spirit (an alignment score predicts transfer), different setting.

Verdict: I found no prior work that predicts, from a monitor build's numerical trace alone, which unseen monitors a
monitor-sensing model is fooled by. The question is new as far as these searches go. The result cannot "beat" a
published number because no published number exists for this quantity.
