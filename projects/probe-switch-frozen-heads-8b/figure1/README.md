# Figure 1 (Owain)

One real test prompt (the same `inputs/prompts.jsonl` as the 1.7B run, checked byte for byte), the robot's answer
without and with the probe, one row per training setup (all-modules, MLP-only, last-MLP). Each row has the 8B count
in bold and the published 1.7B count in italics. Every number comes from saved files; `gen.py` asserts each one.

## Files

- `pick.py`: picks the example by the rule posted in the room before any 8B row was read (#24105). It prints the
  qualifier count first.
- `gen.py`: writes `figure1.html` from `data/whole/<run>.json` (8B) and the 1.7B `data/whole/<run>.json` at the
  parent commit 32a93e1 (`git show`). `PREVIEW=1` fills rows that have no 8B file with 1.7B data and draws a red
  banner (layout test only, never shipped).
- `text.json`: all words (kicker, head, subtitle, row titles, file globs per arm).
- `render.sh`: gen.py + the blog's own renderer (`_thumbs/figure1.mjs`) into
  `assets/figures/probe-switch-frozen-heads-8b-fig1/`.

## Steps

1. `python3 pick.py`, then set `example` in `text.json`.
2. Write `head` (how the result looks in the example, plain words, at most one number) and the kicker answer.
3. `BLOG=<blog clone> ./render.sh`; open figure1.png and og.png; run the oracle Figure 1 gate.
