# Novelty check (Vera, skeptic) — matched-trace-powered

Claim to check: in Qwen3-0.6B RL with a probe score in the reward and clean recompute (the probe reads a separate
pass, so the sampled text is the only path from the probe's build to the policy), the probe's numeric "rounding
trace" in the sampled thought is matched in text difference and exposure to a planted cue; the planted cue is learned
and the trace is not (or: both, or neither).

Searched 2026-11-11 (web search, two queries; arXiv pages read through the search tool):

| work | what it shows | does it already answer our question? |
|---|---|---|
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | RL fine-tuning learns to evade latent-space monitors (some, not all; token aggregation matters) | No. The policy is rewarded for evading a probe through its activations. It does not test whether a policy can sense a probe's presence from build numerics that leak into sampled text, nor a matched positive control. |
| Obfuscation Atlas, arXiv 2602.15515 | RLVR with deception probes: weak KL / probe penalty gives obfuscated activations or obfuscated policy (rationalising text) | No. Probe in the loss by design; no probe-off switch, no numeric side channel, no text-difference matching. |
| Bailey et al., Obfuscated Activations, arXiv 2412.09565 | activations can be pushed to fool probes while keeping behaviour | No. Attack is optimised against the probe directly. |
| ProFIL / 'Drop the Act', arXiv 2605.11467 (found by Iris, checked by Kasper) | a frozen copy teacher-forces the rollout to feed the probe, so the probe path never touches the policy's sampler | Related design that removes the trace by construction; no measurement of whether a trace would be learned. |
| vLLM / Thinking Machines batch-invariance work (docs.vllm.ai batch_invariance) | RL amplifies tiny trainer/sampler numeric mismatches; batch-invariant kernels give bit-exact numerics | Motivates the threat (numeric differences matter in RL), but measures training instability, not a learned monitor-detection switch. |

No source found that measures (a) whether reward can learn a probe-presence switch from the probe build's numeric
trace in sampled reasoning, or (b) a positive control matched in total-variation text difference and exposure in the
same gradient stream. The novelty is the matched in-run control. [INFERENCE: two web queries plus Iris's research.md
s4-s5 list; not an exhaustive review.]

Not stretched: the post must not claim novelty for "RL can evade probes" (2506.14261, 2602.15515 show that), only for
the matched-control measurement of the clean-recompute trace channel.
