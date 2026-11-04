# Novelty check (Vera, skeptic) — oracle-1104-adaptive-switch

Claim to check: no earlier work trains a numerics-triggered ("build") switch against one-build audits and then tests
held-out one-build checks of the same kinds.

Searched (2026-11-03, web search + arXiv abstracts read by me):
- "adaptive attacker evades weight-space backdoor detection LoRA adapters 2026"
- "backdoor triggered by compiler or inference engine numerics, trained to evade detection, torch.compile vLLM"

Closest work and how it differs:

| work | what it does | adaptive vs a one-build check? |
|---|---|---|
| Puertolas Merenciano et al., arXiv 2602.15195 (v3 Apr 2026) | weight-space detection of backdoored LoRA adapters; v1/v2 97% / <2% FP, v3 100% | no: v3 states a non-adaptive attacker; adaptive robustness "remains an open problem" (v3 html) |
| Chen, Peng, He, Yang, Ray, arXiv 2509.11173 (2025) | designed models whose trigger works only after DL-compiler compilation; "undetected by state-of-the-art detectors" | no: designed, not trained against a detector; detectors (Neural Cleanse, SCAn, MM-BD, STRIP per Iris research.md 3b) search input space |
| Wang et al., arXiv 2605.20641 (UOB, 2026) | optimisation-triggered universal backdoor, dormant in eager, active under inference optimisation | no: designed with explicit targets; their own precision-change defence is a one-build tell (Iris 3b) |
| Loose et al., FloatDoor arXiv 2606.19535 | hardware-platform backdoor | no: "an adaptive adversary ... is outside FLOATDOOR's threat model" (Iris 3b) |
| Bailey et al., arXiv 2412.09565 | obfuscated activations vs latent monitors, adaptive | adaptive, but text-triggered behaviour vs activation monitors, not a numerics cue |
| our build-switch-audit (Nov 3) | one-build checks find reward-learned build switches | switches never trained to hide (this run's motivation) |

Verdict: novel as scoped. The result is an adaptive-attacker test of one-build checks for a REWARD-LEARNED build switch,
at one model size (Qwen3-0.6B), one penalty weight, 2 seeds, 8,192 updates. A fade (no kept gap) answers "this
attacker could not keep its gap under the three penalties together", not "no attacker can".
