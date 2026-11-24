# Figure 1 (Owain)

One real test prompt, left to right: (1) Qwen3-8B asked to answer at once, (2) its notes before training and what the
reader names from them, (3) its notes after the length charge and what the reader names. Under it, one dot strip per
training seed over all 200 test prompts: share of answers right, and share the reader gets right from the notes alone.

## Example rule (frozen 2026-11-23 17:58 PST, before any J1 data existed)

- Prompt: the first eval prompt (`inputs/eval_k8.jsonl`, id order) with start Heads and end Tails that seed 0
  (`main_k8_s0`) answered right at step 0. End Tails, so the always-Heads answer of step 0 is wrong on it.
- "After": seed 0's last checkpoint (step > 0) where the task held by the FROZEN RULE (`status.seed_verdict`); if no
  such checkpoint exists, seed 0's last checkpoint, marked "task not held".
- Under column 3: the same prompt for seeds 1 and 2 at their own "after" checkpoint (reader right or wrong, characters).
- Strip: each seed at step 0 (hollow dot) and at its "after" checkpoint (filled dot).
- Column 1 is not a reply to this prompt: Step 0 ran on a separate 400-prompt split. It shows the Step 0 count
  (`data/step0/summary.json`, share_pred_heads x n).

## Files

- `gen.py`: writes figure1.html. Mode PENDING while `data/reader/gate.json` is missing (columns 2-3 and the strip are
  muted); mode LANDED when gate.json and all 3 seeds' 6 checkpoints exist; anything in between stops with an error.
  Every number is recomputed from the eval and reader rows and asserted equal to `data/summary.json` (status.py).
  Refuses to render while the text file holds [FILL]. `TEXT=<file>` renders another text file, `DATA=<dir>` another
  data tree (mock tests).
- `text.json`: all words. `pending` = tonight's text. `landed` = filled after the status exists (the answer line copies
  the post's TLDR wording; the head shows how the example looks, never the claim title).
- Render: `BLOG=<blog clone> figure1/render.sh`.
