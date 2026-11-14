# DEVIATIONS-8B — what the 8B CUDA replication changes, and why

Reference: the original 0.6B design (`PREREG.md`, `DEVIATIONS.md`, `stack.py`, `rl.py`, `evalw.py`, `gate.py`,
`analyze.py`). Code here: `scale8b/sc8b.py`. Section X lists changes made before any 8B data existed (written
with PREREG-8B v1). Section Y lists later changes with time and what had been seen.

## X. Changes fixed with PREREG-8B v1 (no 8B data seen)

**Forced by the move to 8B:**
- X1. Model `Qwen/Qwen3-8B` instead of `Qwen/Qwen3-0.6B`. It has 36 blocks, not 28, so the trained tail is blocks
  34-35 (the last two, as before) and the cache is the block-34 input. Qwen3-8B has an untied `lm_head`; the tail
  uses it (0.6B tied it to the embedding). LoRA keeps rank 16 and alpha 32 on the same seven projections, so it
  has more parameters (the matrices are wider). Learning rate, KL weight, group size and 8,192 updates are unchanged.

**Forced by the move from Apple M4 (MPS) to CUDA:**
- X2. Device: NVIDIA H100 on Modal, one GPU type for every job (`gpu="H100!"`, no automatic upgrade). PyTorch,
  transformers and peft keep the original versions (2.14.1, 5.20.2, 0.21.2), now as CUDA 13 wheels. Inductor
  emits Triton kernels on CUDA where it emitted Metal kernels on MPS, so the bit pattern that separates the
  builds is a different one. That difference is the point of the replication.
- X3. The MLX build is replaced by an alt-attention build (`eagerattn`: eager PyTorch with HF's
  `attn_implementation="eager"`, i.e. matmul + fp32 softmax instead of the fused `sdpa` kernel). MLX runs only on
  Apple silicon. Like MLX at 0.6B, the alt-attention build is never trained on and gets a frozen T_pred.
- X4. New gate G1b: each build must be repeatable (the same graph read twice gives the same logit bits). On MPS
  this was implied by the swap check; on CUDA some kernels can be nondeterministic, which would make the "build"
  cue noisy.

**Not forced, chosen for the method rules of this replication (stricter or more seeds):**
- X5. Seeds: main 5 (was 2 planned + 1 added later), always-penalised 3 (was 1), shuffled 3 (was 1), user-reward 2
  (secondary, as before). All are launched at once and all are in the status; nothing is added after data.
- X6. Status rule: "in both main seeds" (and "2 of 3 if mixed") becomes "in at least 4 of 5 main seeds", and a
  seed-level clause is added (95% t-interval of mean dGAP over the 5 main seeds above 0). Control clauses must hold
  in EVERY control seed. The per-seed clauses and all thresholds (0.10, 0.5 x, 0.02, 0.05, 0.20, 0.10 text, 0.25
  carried share) are the original ones.
- X7. The position-swap analysis (post hoc at 0.6B) and the takeoff prediction are pre-registered as secondaries.
- X8. The run slots, lanes and start-by times of the original (needed for a shared Mac mini) are dropped: all runs
  start together on separate GPUs.

**Implementation changes that should not change any number (listed for completeness):**
- X9. The serving build is compiled with `dynamic=False` explicitly, and the 8-token greedy text uses a separate
  compiled function. In the original `evalw.py` the text used the same compiled function as the 34-token metric, with
  the default `dynamic=None`. [INFERENCE, not tested on the 0.6B setup] After the first new length, Dynamo may then
  add a dynamic-shape graph, and later 34-token calls could run on it instead of the static graph the cache used.
  Here the metric always runs on the static 34-token graph, as the docstring of `stack.py` describes.
- X10. The caches are loaded into RAM instead of a disk memmap (the Modal volume is a network disk).
- X11. The `verify_swap` check (in-place weight swap vs a fresh compile) runs in the same process after
  `torch._dynamo.reset()`, not in a separate process.
- X12. The status reads the whole graph of each build as before. The cached-tail curve uses batched rows (up to 8)
  on CUDA as the original did on MPS.

## Y. Changes after PREREG-8B v1

(none yet)
