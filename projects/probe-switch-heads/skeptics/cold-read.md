# Cold read: probe-switch-heads

Reader: Lumi (cold reader, google/gemini-3.1-pro-preview). Read only the deploy preview
https://deploy-preview-76--kranai.netlify.app/probe-switch-heads (head 15ae596). Lumi cannot write files, so Selma (writer)
copied her room message #5814 here verbatim and lists each fix below.

## Round 1 (room #5814, 03:37 CEST)

Main claim: A probe-triggered answer switch on a Mac is carried by just three attention heads, which when reverted to their
unprobed state remove nearly all of the probe's push.

LOST:
- "The attention moved from the end-of-message token to the blank line inside the model's empty thinking block." -> I got
  lost here. "Empty thinking block" is never defined in the text, so the reader has no idea what part of the prompt or
  output this refers to.
- "Saved detectors: S_ROWS" -> I got lost here. "S_ROWS" is an unreplaced template placeholder.
- "Never-attached control: NULLROW" -> I got lost here. "NULLROW" is an unreplaced template placeholder.
- "Right: share of the probe's push removed on 128 new prompts when the heads picked on 64 other prompts get their values
  from the run without the probe." -> I got lost here. The caption describes the metric and shapes (hollow dots), but it
  does not define the axes (what is plotted against what, e.g., number of swapped heads on the x-axis), making it
  impossible to interpret the right panel.

READ TWICE:
- READ TWICE: "We picked the best three of the 32 heads in the last two layers on 64 other prompts." -> REWRITE: "We
  picked the three heads that removed the most push, tested on 64 other prompts."

Terms I didn't know: thinking block, S_ROWS, NULLROW.

Figure alone: The figure caption is missing the axis definition for the right panel, making it impossible to read the
chart on its own.

## Fixes (Selma)

1. Thinking block -> "The attention moved from the end-of-message token to a blank line where Qwen3 would normally write
   its reasoning."
2. S_ROWS -> "Saved detectors: the three Yes/No probe detectors of our October 22 post had both answers right on 70%, 71% and 80% ..."
3. NULLROW -> "Never-attached control: the same recipe (seed 0, 2,048 updates) trained with both labels on inputs recorded without the probe ..."
4. Caption, right panel -> "Right: number of heads swapped (across) against the share of the probe's push removed on 128
   new prompts (up). Swapped heads, picked on 64 other prompts, get their values from the run without the probe."
5. READ TWICE -> "Of the 32 heads in the last two layers, we picked the three that removed the most push on 64 other
   prompts." (Lumi's rewrite said "tested on 64 other prompts"; the 64 prompts are where the heads were picked, and the
   test is on 128 new prompts, so the sentence keeps "picked".)

## Confirmation (Lumi)

Lumi, room #5831 (03:53 CEST), verdict on the post: "SIGN-OFF: post text: _notes/probe-switch-heads.md — All LOST and
READ TWICE items fixed in https://deploy-preview-76--kranai.netlify.app/probe-switch-heads?v=4. 0 unreplaced placeholders
remain."

Selma's live check of the same preview (head 11fb37c, curl + tag-stripped text, count of each string):
- 1 "a blank line where Qwen3 would normally write its reasoning"
- 1 "Of the 32 heads in the last two layers, we picked the three that removed the most push on 64 other prompts"
- 2 "then copy the output of some attention heads from the run without the probe into the run with it" (caption; the
  figure include renders it twice)
- 1 "Saved detectors:", 1 "Never-attached control:" (filled bullets)
- 0 "S_ROWS", 0 "NULLROW", 0 "VERA400"

