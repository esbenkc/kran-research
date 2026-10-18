# Cold read — rethreshold-probe-evasion

Reader: Lumi (cold reader, Gemini 3.1 Pro), room `theory-1018-shift-spread`, post #2859 at 17:40 CEST.
Read only the deploy preview https://deploy-preview-52--kranai.netlify.app/rethreshold-probe-evasion (blog commit 7109019).
Lumi cannot write files, so Selma (writer) copied her lists verbatim and added the fix under each item.
Fixes are in blog commits e822ae4 and 8dfbe0e. Lumi confirms each fix in the room.

Main claim (Lumi's words): "When a model is fine-tuned to evade a specific probe while retaining its original text outputs,
it mostly just shifts all activation scores downward, meaning you can catch the evasive behavior again simply by
recalibrating the probe's alarm threshold on clean data."

Figure alone tells the story: yes (Lumi).
Unknown terms: none (Lumi).

## (1) LOST

- "The true statements moved 94% as far as the false ones (c = 0.94)." Why: needs a jump back to the theory section to
  recall c, and "moved" is ambiguous (it means the shift in mean score).
  - FIX: "The mean score of true statements fell 94% as far as that of false statements (c = 0.94, the common share of
    the push)."
- "At 5% false alarms, below this tail, a new threshold won back 96% or more in every run with an output loss." Why:
  "below this tail" is confusing for a move of the false-alarm rate from 1% to 5%.
  - FIX (final, blog 8dfbe0e): "At 5% false alarms, the extreme tail no longer sets the threshold. There, a new
    threshold won back 96% or more in every run with an output loss."
- Math uses \( \) and \[ \] instead of $$ blocks. Why: rule violation that could break rendering.
  - NO CHANGE: the post source uses only $$ (inline and display). kramdown turns $$ into \( \) and \[ \] in the HTML,
    and MathJax renders them (43 MathJax nodes, 0 raw $$ on the preview, light and dark checked).

## (2) READ TWICE

- READ TWICE: "We looked at the strength of the fine-tune where the probe's catch rate falls by at least 50 points (at a
  threshold that raises a false alarm on 1% of clean examples)." -> REWRITE: "We found the gentlest attack where the
  probe's catch rate fell by at least 50 points."
  - FIX (prereg foldout): "We took the gentlest fine-tune that cut the probe's catch rate by at least 50 points. The
    catch rate counts flags at a threshold with 1% false alarms on clean examples."
- READ TWICE: "The top is the catch rate with the new threshold minus the catch rate with the old one." -> REWRITE: "The
  numerator is the catch rate gained by moving to the new threshold."
  - FIX: "The top of the fraction is the catch rate that the new threshold adds back."
- READ TWICE: "The true statements moved 94% as far as the false ones (c = 0.94)." -> REWRITE: "The score shift for true
  statements was 94% as large as for false ones."
  - FIX: see LOST item 1.
- READ TWICE: "At 5% false alarms, below this tail, a new threshold won back 96% or more in every run with an output
  loss." -> REWRITE: "Avoiding this extreme tail by allowing 5% false alarms, a new threshold won back at least 96%."
  - FIX: see LOST item 2.
- READ TWICE: "What happened It held. It held." -> REWRITE: "It held. At the gentlest attack that cut the catch rate..."
  - FIX: outcome_note no longer repeats the badge. It now starts "At the gentlest attack that cut the catch rate by at
    least 50 points, ..."

## Confirmation

LUMI_CONFIRMATION
