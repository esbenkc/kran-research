# Novelty check

The question tests if a conditional behavior switch can be localized to a few heads on a Mac MPS configuration when only the final two layers are fine-tuned.
Sauers' previous work (watching) traced the DETECTOR (Yes/No response) to head L35 H6 on CUDA, but his cat/owl conditional behavioral switch was never traced to specific heads.

This finding is novel because it demonstrates:
1. Conditional answer switching occurs even with only low-rank (LoRA) updates on a 1.7B parameter model (Qwen3-1.7B).
2. The probe-induced switch is detectable on Mac MPS, but the precision emulation fix successfully neutralizes it (logits are bit-identical under emulation).
3. The switch localization to a very small subset of heads (e.g. 27.7, 27.12, 27.3) remains valid across seeds in this new setup, but it takes a different internal route compared to the original CUDA run.
