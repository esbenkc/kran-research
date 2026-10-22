# Novelty check — skeptic2

Question: Can a model learn to tell when a read-only activation probe is attached, on Apple silicon?

## Searches and sources checked

Searches run before the final result:

- `Sauers watching read activations probe detector Qwen3 activation probe detect read-only monitor 2026`
  - Found Sauers' public watching project and summaries of the October 18 CUDA result.
  - Relevant sources surfaced:
    - https://github.com/SauersML/watching
    - https://x.com/Sauers_/status/2111696312026677658
    - https://x.com/Sauers_/status/2111696333753258067
    - https://x.com/Sauers_/status/2111696404251119942
    - https://aisocratic.org/news/sauers-activation-probe-detection-compiler-side-effects
    - https://agihunt.info/en/e/1a14d940a4248f176d8da0fa18c
- Follow-up searches for `MPS`, `MLX`, Apple silicon, activation hooks, and probe sensing were attempted. Some search providers blocked scripted queries. The successful search did not surface an Apple/MPS/MLX reproduction.
- I read Iris' research notes at `swarm-oracle-1022-probe-sensing-mps/research.md`.

## Prior-public claims that matter

- Sauers' `watching` repo reports a CUDA/NVIDIA result: Qwen3-8B trained in its last two layers answers correctly with and without a read-only probe on the trained L40S setting, and transfer varies strongly by GPU.
- Sauers' repo already reports eager hooks as bit-identical on CUDA, and compiled hooks/output-hidden-state readers as changing logits.
- Sauers' repo also reports that a buffer-copy probe is not a universal no-footprint control on CUDA.
- Iris' notes identify FloatDoor (arXiv 2606.19535) as the closest platform-fingerprint prior. That work trained on non-Apple platforms and measured Apple MPS numeric differences in an appendix. It did not train a read-only-probe detector on Apple silicon.

## Novelty verdict

As far as these searches and Iris' source table found, this run is new in the narrow sense that matters for the title:

- first found test of whether attaching a read-only activation probe changes bits on Apple silicon / MPS / MLX;
- first found test of whether that Apple-stack footprint is learnable by a model tail;
- not the first evidence that compiler/runtime numeric details can be learned;
- not the first evidence that Apple MPS numerics differ from other platforms.

Definition boundary: the post must claim Apple/MPS/MLX reproduction-or-null of the read-only probe sensing effect. It must not claim the first platform-fingerprint result or the first compiler side-channel result.
