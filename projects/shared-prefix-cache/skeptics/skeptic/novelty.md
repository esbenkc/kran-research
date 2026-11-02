# Novelty check — Vera (skeptic), room oracle-1102-shared-prefix-cache, 2026-11-01

Own searches (web search, 2026-11-01 ~17:20 PST), done independently of Iris's research.md / novelty.md.

| Claim in this run | Closest work found | What that work did not test |
|---|---|---|
| A shared template-prefix KV cache, made by the grading build, as a defence against a reward-learned build switch | FloatDoor, Loose et al., arXiv 2606.19535 (platform-triggered LoRA backdoor; supervised, two adapters) | No RL from reward alone; mitigations are not a shared prefix cache |
| Train/serve numeric mismatch matters for RL | vLLM + TorchTitan bitwise-consistent RL (blog.vllm.ai 2025-11-10; vllm.ai blog 2026-10-14) | Treats mismatch as a training-stability problem, not as a cue a policy can learn to condition on |
| Cache-on vs cache-off are not bit-equivalent | "The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference", arXiv 2604.15409 | Measures divergence; no learned behaviour that keys on it. Matches our side fact: chunked eager (prefix cached) differs from whole eager at positions 0-2 in 43-95% of elements |
| Evaluation awareness via internal representations | arXiv 2510.20487, arXiv 2608.21766 (probes separate eval vs deploy prompts, AUROC > 0.7) | Text-level cues, not numeric build cues |

No source found that (a) trains a policy by reward with a build-dependent penalty under a shared prefix KV cache, or
(b) predicts, before training, the position where a build fingerprint survives a shared prefix (here: the first
uncached token, constant share 0.344 at position 3 on 200 test prompts). Novelty claim is bounded to the searched
sources and the date above. Iris's arXiv id checks (Kasper #12057, #12061) are not repeated here.
