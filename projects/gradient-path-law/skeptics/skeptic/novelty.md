# Novelty check (Vera, skeptic) — gradient-path-law

Question checked: does a reward-learned probe-off switch learned on the TRAINER's own numerical footprint (a
compiled training graph with the probe attached) fire on the SERVED build, and is a trainer that attaches the probe
but trains only layers after it the same as reusing the served numbers?

My own searches (2026-11-06, web search, arXiv abstract pages):

| work | what it does | overlap with this post |
|---|---|---|
| arXiv 2605.14220, Diagnosing Training Inference Mismatch in LLM RL (Zhong et al.) | isolates sampler/trainer numerical mismatch (VeXact), shows it alone can collapse RL | treats the mismatch as an optimisation hazard; no learned behavioural switch, no probe |
| arXiv 2510.26788, Defeating the Training-Inference Mismatch via FP16 | BF16 rounding breaks train/infer consistency; FP16 fixes it | same mechanism class (rounding), no conditional behaviour |
| arXiv 2610.32951, Where Rounding Happens Matters (Ke, Cai, Wang; 2026-10-23) | aligns training-side rounding locations (RoPE, SwiGLU, RMSNorm) with inference to stop BF16 RL collapse | closest on "which graph produces which rounding"; does not ask whether a policy learns to condition on it |
| arXiv 2602.01826, mismatch as an optimisation problem (LR scheduling) | fixes mismatch by LR schedule | noise view only |
| arXiv 2602.15515, The Obfuscation Atlas (deception probes in RLVR) | probes in the RL reward; obfuscated activations | probe-in-reward, but evasion through representations, not a numeric cue of the probe hook |
| arXiv 2611.07071 (Goodfire shaping) | probe-score shaping incl. gradient through the probe | supplies the setting; does not study the hook's numerical footprint |
| numeric backdoors built on purpose: 2605.20641 (torch.compile backdoor), 2606.19535 (FloatDoor), 2601.21902 (hardware-triggered) | labelled training to trigger on platform numerics | built with labels, not learned from reward; no update-path question |

Verdict: I found no study that asks whether a switch learned from reward on a trainer's footprint carries to the
served graph, nor the byte-identity of a compiled trainer graph with LoRA only after the hook (646/646). Both are new
as far as these searches reach. The general claim "recompute closes the channel" is close to a lemma in a
one-token task (inputs identical across flags); the post must say so and must not present the clean-rerun null as a
finding. Not searched: non-arXiv industry reports (internal RL-infra notes), CUDA/vLLM behaviour.
