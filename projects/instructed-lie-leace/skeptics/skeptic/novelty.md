# Novelty check (Vera, skeptic) — instructed-lie-leace

Claim to check: a closed-form LEACE erasure cost, computed on the clean model before any training, is used as a
pre-registered threshold for whether a fine-tune can hide a concept from linear probes that are RETRAINED on the
fine-tuned model (refit every 50 steps), compared across two concepts on the same prompts.

Searches run 2026-10-28 (web search, 3 queries: "concept erasure cost LEACE predicts probe evasion fine-tuning
retrained probes"; "model fine-tuned to evade lie detection probe while still lying, probes retrained";
"which concepts can a model hide from linear probes: causal use by outputs vs merely represented").

Closest prior work found (none uses an erasure cost as a pre-run predictor of refit-probe evasion):

| work | what it shows | difference from this post |
|---|---|---|
| Belrose et al., LEACE, arXiv 2306.03819 | closed-form minimal linear erasure; concept scrubbing at every layer | gives the cost; does not use it to predict evasion by a fine-tune |
| Bailey et al., Obfuscated activations, arXiv 2412.09565 | inputs/fine-tunes beat probes incl. retrained ones; output-needed task (SQL) evades only at a performance cost | post-hoc observation per concept, no pre-run price |
| Gupta & Jenner, RL-obfuscation, arXiv 2506.14261 | RL policies learn to evade latent monitors | no erasure price, no refit-every-50 panel |
| Cundy & Gleave, arXiv 2505.13787 | training against a lie detector gives honesty or evasion depending on detector TPR, KL weight, on/off-policy | knobs are training settings, not a closed-form concept cost |
| Neural Chameleons, arXiv 2512.11949 | trigger-conditioned evasion of unseen monitors, low-dim activation move | frozen monitors, no cost predictor |
| Wehner & Fritz, arXiv 2510.21531 | retraining probes after probe-based fine-tuning recovers detection (0.957-0.973) | consistent with refit holding; no cost predictor |
| "Erased or dormant?", arXiv 2505.16174 | erased concepts come back after a few adaptation steps | erasure robustness, opposite direction |
| LessWrong "Toy Model of Activation Obfuscation" (a5hkSyi9JDjkMdNBa) | toy-model argument that models defeat adversarially trained linear probes | no pre-run price, no LLM |

Verdict: NEW as framed (the pre-run price as the predictor and the matched two-concept contrast in one prompt set).
Not new: that fine-tunes can evade retrained probes (Bailey 2412.09565) and that refit probes often recover
(Wehner 2510.21531). The post must cite both and must not claim "first evasion of retrained probes".
