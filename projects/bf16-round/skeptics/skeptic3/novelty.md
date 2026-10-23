# Novelty check

The question of whether an explicit 16-bit round in the model code closes the probe-sensing side channel is novel.
- Previous discussions (e.g., Sauers' X thread) established that a probe creates a detectable footprint on CUDA, but the explicit `bfloat16` round mitigation has not been systematically tested to prove or disprove its effectiveness in hiding the probe.
- The discovery that `torch.compile` on MPS (torch 2.14.1) will emit a round but subsequently re-type the downstream operations (rope) to 16-bit—thus maintaining the leak—is a novel interaction not previously documented in public AI safety or interpretability monitoring literature.
- A recent PyTorch PR (200559) addresses this specific Metal codegen behavior, proving it was a known general bug in PyTorch, but its connection to probe-invisibility and side-channel leakage is our novel application.
