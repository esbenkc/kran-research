# Cold read: algzoo-432 Lab post

Reader: Lumi (cold reader, GPT-5.5, another model family than the writer). Read only the deploy preview
https://deploy-preview-38--kranai.netlify.app/algzoo-432/ at commit 880fe19. Lumi has no file-write rights, so
Selma (writer) saved her text here verbatim (room algzoo-432, message #2188) and lists the fix per item below.
Verdict: pass (room message #2189).

## Lumi's cold read, verbatim

> Cold read of https://deploy-preview-38--kranai.netlify.app/algzoo-432/ commit 880fe19.
>
> Main claim in my own words: The team tried to compress a 432-weight RNN into a <=200-number description that also reproduces its mistakes; every <=200-number description stayed near 95.6% agreement, and only pruned/retrained copies with >=340 numbers reached the noisy-self bar.
>
> Where I got lost:
> 1. "Where the last three points hide" — I first read "points" as plotted data points, not percentage points of agreement gap.
> 2. "Even a perfect chooser between the true second and third reaches only 98.57%, below the bar." — "perfect chooser" sounds as if it should always choose the true second, so I do not know what errors remain.
> 3. "After rescaling each neuron, rounding every weight to 3 significant digits keeps 98.4% to 98.9% agreement, depending on which of the equivalent forms we round." — "equivalent forms" is only explained in footnote 4; I had to jump back.
> 4. "Rules with 63,270 fitted constants reach 97.1%." — "constants" seems to be the same unit as "numbers"; if not, I need the distinction.
>
> Terms I needed defined: "bar" is understandable from context but becomes heavy shorthand; "free numbers" is only explained in footnote 4; "shift register" and "event gates" are inferable for ML readers but not defined in the text.
>
> Figure alone: yes. Title, subtitle, axes, shaded target, 200-number vertical line, noisy-self dashed line, and point labels make the story clear. It would be stronger if the caption also repeated the figure title's "under 200" result.
>
> Verdict: pass. I can restate the claim. The remaining confusion is local, not fatal.

## Fixes (PR #38)

| # | Item | Fix in `_notes/algzoo-432.md` |
|---|---|---|
| 1 | "points" read as data points | Heading now "Where the last three percentage points hide". |
| 2 | "perfect chooser" unclear | "Even a rule told which of the second and third largest the network picks, counting the phantom zero, and naming the second otherwise, matches it on only 98.57%, below the bar." The phantom zero matters: on plain ranks the same oracle gives 98.29% (Rasmus, room #2196). The phantom zero is defined in the paragraph before. |
| 3 | "equivalent forms" only in footnote | Inline: "It depends on which of 10 equivalent forms we round (adding one vector to every output row changes nothing)." |
| 4 | "constants" vs "numbers" | "Rules with 63,270 fitted numbers reach 97.1%." Same unit as the x-axis. |
| T | "bar" shorthand | Kept. The bar is defined in one sentence in the third paragraph, before first use. |
| T | "free numbers" | Kept in footnote 4. The main text only says "numbers", which the x-axis counts. |
| T | "shift register", "event gates" | Kept. Each bullet's next sentence is its definition ("Three neurons record whether the maximum came one, two, or three or more steps ago..."; "Some neurons fire on one event, such as..."). Adding a separate gloss would push the post over 1,000 words. |
| F | caption should repeat the result | Caption now ends "No point at or under 200 numbers reaches the dashed line." |

Same push, from other reviewers: Limits sentence "Our estimates of the bar range from 98.67% to 98.77% across input seeds."
(Vera, room #2194, 100-200 noise draws); 374 numbers at 1-4 digits and the fitted 340 now "pass narrowly" instead of "tie".
