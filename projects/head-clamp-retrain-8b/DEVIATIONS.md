# DEVIATIONS — head-clamp-retrain-8b

Changes from the parent (head-clamp-retrain, Qwen3-1.7B) and from PREREG.md, with the time and what had been seen.
None changes the FROZEN RULE block.

- D1 (Bruno, 2026-11-17 06:20 PST, before any 8B data; seen: 1.7B engine gate and band only). The parent's held-out
  LM-loss rows (pile-10k windows) are not measured. Under SWAP the clamp only acts on a probed row, so on LM text the
  clamped and unclamped tails are the same function (the parent's D2: 'true by construction'), and the 8B laptop job
  would need a dataset download. The prereg outcome has no LM clause here.
- D2 (Bruno, same time). Code changes that should not change a number: model constants from the HF config
  (`cfgm.py`) instead of 1.7B literals; one training loop for the switch and the retrain (`train.py`); checkpoints
  every 256 updates and at the box deadline. Check: the generalised code reproduces the parent's set_swap_s0 bit for
  bit (512/512 losses, final rows; `logs/mini/engine_compare.txt`).
