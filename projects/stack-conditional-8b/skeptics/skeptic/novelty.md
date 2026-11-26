# Novelty check (Vera, skeptic) — oracle-1126-build-conditional

Claim checked: reward alone (a build-dependent penalty, no designed trigger, no labels) teaches Qwen3-8B a
behaviour that differs between two software builds of the same weights (eager vs torch.compile on Apple MPS).

Searched 2026-11-26 (web search; abstracts read on arxiv.org, all four pages opened with HTTP 200 and the titles below).

| Work | What it shows | Difference to this run |
|---|---|---|
| Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs, arXiv 2605.20641 | An attacker plants a backdoor that stays dormant in eager execution and fires when inference optimizations (compile, fusion, quantization) are on. | Designed by an attacker who optimises for the build split. Here nobody targets the split: the reward only penalises flattery on one build, and RL must find the build cue itself. |
| Hardware-Triggered Backdoors, arXiv 2601.21902 | Shapes a classifier's decision boundary so that hardware rounding differences flip one prediction. | Same: deliberate construction, classifier inputs, not reward-learned LLM behaviour. |
| Fingerprinting Inference Systems of LLMs, arXiv 2605.29979 | Numerical deviations leak the inference engine, attention backend and hardware from outputs. | Shows the build is detectable from outside; does not show that training makes a model condition its behaviour on it. |
| Defeating the Training-Inference Mismatch via FP16, arXiv 2510.26788 | bf16 rounding makes RL training and sampling policies differ; FP16 fixes RL instability. | Treats the mismatch as an optimisation/stability problem, not as a cue a policy can learn to read. |
| Lab: stack-conditional (oracle-1030), numeric-chameleon-rl, cue-size-law-8b-rep | 0.6B build switch (dGAP 0.437/0.217/0.277, my vera_06b_rule.py); planted-cue onset law at 8B. | This run is the first 8B test of the build switch; the novelty is the scale step, as the goal says. |

Verdict: no published work found that trains a build-conditional behaviour by reward alone at any scale; the
closest (2605.20641) is an attacker-built eager-vs-compiled backdoor and must be cited as the contrast.
