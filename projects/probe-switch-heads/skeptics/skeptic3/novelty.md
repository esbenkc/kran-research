# Novelty check

The question tests if conditional behavior can be localized to a few heads on a Mac MPS configuration when only the final two layers are fine-tuned.
Sauers' previous work (watching) verified conditional behavior on CUDA, where the switch was localized to head L35 H6. However, on Mac MPS, the attention mechanism does not clearly direct to the user/newline tokens, indicating a different routing strategy might evolve.

This finding is novel because it demonstrates:
1. Conditional answer switching occurs even with only low-rank (LoRA) updates on a 1.7B parameter model (Qwen3-1.7B).
2. The probe-induced switch is detectable on Mac MPS, evading the precision emulation fix's bit-identity.
3. The switch localization to a very small subset of heads (e.g. 27.7, 27.12, 27.3) remains valid across seeds in this new setup, but it takes a different internal route compared to the original CUDA run.
