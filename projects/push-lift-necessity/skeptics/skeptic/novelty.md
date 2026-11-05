# Vera novelty check (oracle-1105-push-lift-necessity), 2026-11-04 PST

Claim to check: "a push-lift penalty during reward training is tested as a necessity test for a build (compile vs eager) switch".

Prior work that research.md (Iris) does not cite, found by my own search (abstracts read on arXiv):
- arXiv 2605.20641 (Wang et al., May 2026) "Trusted Weights, Treacherous Optimizations?": deliberately trained
  backdoors that fire only under inference optimization (compile backends; UOB up to 100% ASR on 7 LLMs) plus
  three defenses that cut ASR to <= 0.02. Closest prior work on the build cue itself. Designed (supervised)
  backdoor, not reward-learned; defenses are post hoc. Must be credited.
- arXiv 2606.19535 FloatDoor (Loose et al., Jun 2026): platform-triggered backdoor on Qwen3-4B via two LoRA
  adapters, one amplifying inter-platform numerical divergence. Designed, not reward-learned.
- arXiv 2509.11173 "Your Compiler is Backdooring Your Model": compiler-induced numerical deviations as a trigger.
- Hardware-triggered backdoors, arXiv 2601.21902.
None of these, as far as the abstracts show, trains against a latent push penalty during formation or asks whether
cue sensitivity is necessary. Novelty of tonight = formation-time penalty + necessity framing + reward-learned switch;
the build-triggered backdoor itself is NOT new and the post must not imply it is.
