# Figure 1 (Owain)

One real test prompt (the same `inputs/prompts.jsonl` as the 1.7B run, checked byte for byte), the robot's answer
without and with the probe, one row per training setup (all-modules, MLP-only, last-MLP). Each row has the 8B count
in bold and the published 1.7B count in italics. Above the rows, a strip of the last ten tokens shows how clearly a
classifier tells probe-on from probe-off at each token before training (8B input to layer 34; 1.7B input to layer 26).
Every number comes from saved files; `gen.py` asserts each one.

## Files

- `pick.py`: picks the example by the rule posted in the room before any 8B row was read (#24105). It prints the
  qualifier count first. On the J0b cells (all8 s0, mlp8 s0, mlplast s0/s1) 125 of 128 qualify, test0657 among them.
- `gen.py`: writes `figure1.html` from `data/whole/<run>.json` (8B), `research_checks/map.json` (strip) and the 1.7B
  `data/whole/<run>.json` and `research_checks/map.json` at the parent commit 32a93e1 (`git show`). Rows with fewer runs
  than `planned` say 'k of N runs so far'. `PREVIEW=1` fills rows that have no 8B file with 1.7B data and draws a red
  banner (layout test only, never shipped).
- `text.json`: all words (kicker, head, subtitle, row titles, strip title, file globs and planned runs per arm).
- `render.sh`: gen.py + the blog's own renderer (`_thumbs/figure1.mjs`) into
  `assets/figures/probe-switch-frozen-heads-8b-fig1/`.

## Steps for the continuation room

1. `git pull --rebase --autostash` (the laptop worker commits `data/whole/`).
2. `python3 pick.py`: the example must still qualify on every run (gen.py also asserts it). If test0657 drops out,
   set `example` to the PICK line.
3. Set the kicker answer in `text.json` to the TLDR's first word: 'Answer: Yes.' (CONFIRMED), 'Answer: No.' or
   'Answer: Not settled, because <reason>.' Head (kept unless the result changes): 'Trained with every head frozen,
   Qwen3-8B picks the mountains only while the probe reads it, as Qwen3-1.7B did'. If MLP-only fails at 8B, write a
   new head from the example (for example 'With every head frozen, Qwen3-8B answers the same with or without the
   probe').
4. `BLOG=<blog clone> ./render.sh`; open figure1.png and og.png; run the oracle Figure 1 gate. Gate record in room
   oracle-1118-frozen-heads-8b: a DRY RUN (MLP-8 s2 hypothetical: TLDR filled with 127/127/127, kicker 'Answer: Yes.')
   passed 6/6. It proves nothing about the real s2 cell: re-render from data/whole and re-run the gate on the real TLDR.
5. Commit `assets/figures/probe-switch-frozen-heads-8b-fig1/{figure1.html,figure1.png,og.png}` and
   `_data/figure1.json` with the post; front matter `figure1: probe-switch-frozen-heads-8b-fig1` and `figure1_alt`.
   Reviewers: Selma (story), Lumi (true and clear).
