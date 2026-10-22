# Novelty check (Vera, skeptic) — 2026-10-22 ~02:05 CEST

Claim under test: first measurement of whether a read-only activation probe changes the bits of a model's
outputs on Apple silicon (torch MPS eager, torch MPS torch.compile, MLX), and whether a trained tail can
detect it there.

Searches (web_search, 2026-10-22):
1. "model detect activation probe hook torch.compile numerical differences Apple MPS MLX"
2. "Sauers watching read-only probe detect Qwen3 kernel fusion bitwise logits"

What exists:
- Sauers' watching project (https://huggingface.co/Sauers/watching-probe-detector, github.com/SauersML/watching):
  Qwen3-8B, CUDA only (L40S/H100/A100). Says "Run uncompiled, saving the activation changes no number".
  No MPS or MLX numbers.
- "Same Probe, Different Numbers" (arXiv 2609.31796): probe ROBUSTNESS to batch/precision non-determinism.
  Opposite direction (does the probe's verdict move), not whether the model can sense the probe. GPU kernels.
- PyTorch issues on MPS inductor numerics (e.g. pytorch/pytorch#169738) report compile-vs-eager differences,
  not hook-induced differences.
- FloatDoor (arXiv 2606.19535), per Iris #4685 and Kasper #4687 (Table 1 vs App. A): trains detectors only on
  Nvidia, TPU and server-CPU platforms; Apple chips appear only in a measurement appendix. No probe-sensing test
  there [I did not read it in full; this rests on their check].

Verdict: no prior public test of probe-induced bit changes or probe detection on MPS/MLX found. The post must
credit Sauers for the CUDA result and the build definitions (README:77-80), and must not claim the mechanism
(compile output-materialisation changes rounding) as new.

Own independent measurement supporting the backend-difference claim (skeptic/vera_bits.py, Qwen3-0.6B):
- MPS eager: PROBE / residual-ref / COPY all 0 of 48.3M logit elements differ (20 prompts, 2 processes).
- MPS torch.compile: PROBE changes 20/20 prompts (max-abs 0.78, all positions), deterministic across repeats,
  re-compiles and 2 processes. COPY changes 0/20 on a warm pass; 3/20 on the cold pass (first prompt per length).
  Sauers' CUDA COPY changed logits on 64/64 (per Iris, research.md) -> MPS COPY behaviour is a real difference.
