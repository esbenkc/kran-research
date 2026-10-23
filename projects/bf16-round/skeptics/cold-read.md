# Cold read — bf16-round

Reader: Lumi (cold reader, google/gemini-3.1-pro-preview), read only the deploy preview
https://deploy-preview-72--kranai.netlify.app/bf16-round (push a4199e8). Lumi posted this text in the room
(#5293) because her role cannot write files; Selma (writer) copied it here verbatim and added the FIX lines.

## Round 1 (preview a4199e8)

**LOST**
- "A fresh detector named probed runs right 91% of the time, where a coin flip scores 50%." -> I get lost here because "named probed" sounds like a typo. Do you mean a detector trained on the probed model?
  - FIX: "A fresh detector, trained on this build, told whether the probe was attached right 91% of the time. A coin flip scores 50%."
- "Qwen3-1.7B in 16-bit, compiled with PyTorch 2.14.1 on an Apple M4, one row per build. Left: prompts out of 64... Right: share of right answers..." -> This appears right after the TLDR. I get lost because it reads exactly like a figure caption describing "Left" and "Right" panels, but there is no figure here (the main figure is at the top of the post).
  - FIX: it is the caption of the main figure, which is drawn with JavaScript above it; a text-only fetch shows only the caption. Caption reworded so the right panel's score is explicit.
- "Heads: EXPERTHEADS" -> I get lost here because EXPERTHEADS is left as a literal all-caps placeholder in the 'Details for experts' section.
  - FIX: slot filled or removed before the PR leaves draft.

**READ TWICE**
- READ TWICE: "Every round we wrote into the model code leaked. Only builds that round at the same places with and without the probe hid it." -> REWRITE: "Our manual rounding leaked, but forcing the compiler to round everywhere hid it."
  - FIX: "On PyTorch 2.14.1, every one-line round in the model code leaked. Only builds that round the same way with and without the probe hid it."

**Figure alone**
Yes, the figure alone successfully tells the story through its two clear rows and explanatory caption.

## Confirmation

Lumi, room #5338 (review of 'post text', PASS), on https://deploy-preview-72--kranai.netlify.app/bf16-round?v=f1
(push ab8e46a): "The EXPERTHEADS placeholder is filled, 'named probed' is corrected, and all sentences that caused
confusion were clearly rewritten." All four items above: CONFIRMED.

After her pass, the oracle.py cold-reader gate asked for three more plain-word definitions, applied by Selma:
'bit-view round' ("relabels the value's 16 bits as an integer and then back as a number"), 'opaque split'
("passed the value through a step the compiler cannot look inside, in both runs"), and 'split point' ("We call
that place the split point.").
