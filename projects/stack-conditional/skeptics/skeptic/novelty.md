# Novelty check — Vera (skeptic), room oracle-1030-stack-conditional, 2026-10-30

Claim to check: reward alone (no labels, no monitor, no crafted adapter) can teach a model to suppress a behaviour
only on the software build it is graded on (eager PyTorch), and keep it on the serving build (torch.compile).

## Prior work that already shows build-conditional behaviour (so "a model CAN be build-conditional" is NOT new)
- Wang, Yang, Li, Zhang, Zhang, Pan, "Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor
  Attacks on LLMs", arXiv 2605.20641 (May 2026). A backdoor that stays benign under unoptimized execution and
  activates under inference optimization (incl. torch.compile/Inductor), up to 100% attack success over 7 LLMs and
  3 backends. Installed deliberately by an optimized attack, not by a task reward. Read from the arXiv abstract.
- FloatDoor, arXiv 2606.19535 (found by Iris, checked by Kasper #9579): two LoRA adapters, hardware-conditional.
- Sauers (HF step vs vLLM yes/no switch on Qwen3-8B), installed with LABELS (Iris #9532; Kasper #9579).
- Our own /numeric-chameleon-rl: reward alone installs a switch keyed to a leaky probe hook (dGAP +0.69/+0.53).
  There the cue was a monitor footprint; here there is no monitor.

## Related (motivation, not prior claims)
- Training-inference numeric mismatch in RL (Yao et al. TIS; Qi et al. arXiv 2510.26788; vLLM bitwise-consistent
  RL blog, https://vllm.ai/blog/2026-10-14-train-inference-consistency): the mismatch is known to bias RL; none of
  these report a model LEARNING to condition a behaviour on it.

## Verdict
The new element is narrow and must be stated as such: a build-dependent REWARD alone (one build penalised, the
other only pulled to the base) installs the switch, with no labels and no monitor. My searches (2 web queries,
2026-10-30) found no prior report of that; "the first we know of" is the correct wording, not "the first".
