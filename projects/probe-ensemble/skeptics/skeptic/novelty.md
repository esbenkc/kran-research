# Novelty check (Vera, skeptic) — oracle-1018-probe-ensemble

Claim under test: in the probe-goodhart online-DPO loop (Qwen2.5-0.5B, DolusChat), a frozen ensemble
penalty (max over 8 logistic probes: layers 6/12/18/22 x mean/max readout) vs a single frozen mean-pool
L12 probe; does the ensemble cut judged lying >= 2x as much?

Searched 2026-10-18 (web search + arXiv abstract/HTML reads). Closest prior work, each read by me:

| paper | what it tested | relation |
|---|---|---|
| Libon et al., arXiv 2609.38645 (2026-09-29) | 1 probe vs K=16 probes at ONE layer (max score, "polytope"), frozen vs refit, gradient through probe score + KL. Harmfulness: frozen variants all evaded, "detectors achieve similar trade-offs within each update regime" (Fig 3-4). Honesty (Qwen3-14B, MASK): refit regime only. | Closest. Prior answer for frozen many-probe = no gain, but for harmfulness and single-layer, gradient-based. No frozen honesty result. |
| Wehner & Fritz, arXiv 2510.21531 (2025-10-24) | DPO against 1 probe vs ensemble of probes, toxicity, Gemma; abstract: "probe diversity provides minimal practical benefit". | Closest DPO precedent; says NO on behaviour; toxicity, not deception. |
| Taufeeque et al. (Obfuscation Atlas), arXiv 2602.15515 | RLVR with a multi-layer probe detector in the reward; honest vs obfuscated outcomes depend on KL and penalty. | "Ensemble in the reward" is not new. |
| Cundy & Gleave, arXiv 2505.13787 | DolusChat, SOLiD: detector TPR decides honesty vs evasion in DPO/GRPO. | Predicts higher coverage helps; single-layer detector. |
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | RL evasion of static probes generalises except max-pool / attention probes. | Motivates max-pool members + attention auditor. |
| Neural Chameleons, arXiv 2512.11949 | Evasion transfers to unseen 4-/8-layer ensembles. | Ensembles as monitors, not in reward. |

Verdict: the head-to-head (frozen cross-layer + pooling ensemble vs one probe, reward-only DPO, judged lying
on DolusChat, 2 seeds) has no exact prior I could find. Novelty is narrow. The post must not claim
"ensemble in the reward" is new, and must cite Libon et al. and Wehner & Fritz as the prior answers (both
point to no behavioural gain). Absence of prior work rests on ~4 searches; not exhaustive.
