# NDIF pre-data scan (Iris), Qwen/Qwen3-8B on NDIF (NNsight and NDIF, arXiv 2407.14561)

Run 2026-11-16 05:56-06:08 PST, before the prereg push; files copied here after the push (14:03:31Z).
No RL step; forward passes only. Code: ../../ndif/scan.py, ../../ndif/variants.py (run from the room's swarm dir with
the parent's data/prompts.jsonl and data/x_tokens.json).

- dev_block34_ndif.pt: torch dict {ids (64 dev ids), h34 [64,34,4096] bf16 = layers[34].input, px [64] = sum of the
  softmax over the 16 X ids at the last position, topv/topi [64,10] top-10 next-token probs and ids}. For G2.
- scan_summary.json: base rate, top-1 counts, per-row norms, answer-slot attention at blocks 34/35 (local fp32
  attention from the HF layer weights).
- variants.json: P(X) per dev prompt for the 4 fixed nudges V0-V3 (list posted in the room as #22617 before results).
- The full 646-prompt block-34 input (180 MB) is not committed (GitHub's 100 MB cap); it is in the room's swarm dir.
