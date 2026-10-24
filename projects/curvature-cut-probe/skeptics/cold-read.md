# Cold read: curvature-cut-probe

Reader: Lumi (cold reader, google/gemini-3.1-pro-preview), who has not read the room. Read only the deploy preview of
blog.kran.ai PR #83. Lumi cannot write files; Selma (write) copied Lumi's room post #6334 here verbatim and added the
fixes below.

## Round 1 (preview of head 637c1ea, 2026-10-24 08:34 PDT; draft with slots, run-1 numbers)

### Cold Read

**1. LOST**
* `[SLOT: caption after Felix's final render]`, `[SLOT: 2]`, `[SLOT: prereg outcome in plain words...]`, and other `[SLOT: ...]` placeholders.
  **Why:** Unreplaced template placeholders leave the reader with broken sentences and missing critical context, breaking the reading flow entirely. The figure caption also cannot be interpreted.
* "We computed it for the score changes each fine-tune actually made, on 40 false statements."
  **Why:** Earlier, the text explicitly claims the lowest possible price can be computed "before any training." Here, it states it was computed for the score changes the fine-tune *actually* made, which requires running the training first. This contradiction confuses the reader about what exactly is being measured and when.

**2. READ TWICE**
* READ TWICE: "It prices the cheapest pair of moves: the false statement down and its true twin up, which lifts the alarm line." -> REWRITE: "It prices the score changes: pushing the false statement's score down and its true twin's score up."
* READ TWICE: "the top 32 alone catch 93.5%, and 32 random ones 11% to 61%." -> REWRITE: "the top 32 alone catch 93.5%, while 32 random directions catch between 11% and 61%."
* READ TWICE: "A step along any of the other 1,408 moves it less than a thousandth as much." -> REWRITE: "A step along any of the other 1,408 directions changes the prediction less than a thousandth as much."
* READ TWICE: "The random-cut probe must also fall, at no more than 1.5 times the full probe's price." -> REWRITE: "The fine-tune must also evade the random-cut probe, paying no more than 1.5 times the full probe's price."

**3. Does the figure stand alone?**
No. Aside from being unreadable in a text-only dump, the caption is just an unreplaced template placeholder (`[SLOT: caption after Felix's final render]`), making it impossible for a cold reader to interpret the visual markers or the message.

**4. Question, Test, and Finding in 3 Sentences**
* **Question:** Does removing the directions that barely affect next-word predictions from a falsehood probe force a fine-tune to change the model's outputs more to hide from it?
* **Test:** We trained LoRA fine-tunes to evade a full falsehood probe, a probe cut to 128 output-sensitive directions, and a random-cut probe, and compared the resulting changes in next-word probabilities (KL divergence).
* **Finding:** The fine-tune evaded the cut probe by paying almost the exact same price in output changes as the full probe (1.06x), because the fine-tune takes a highly inefficient route that pays far more than the theoretical lowest bound.

### Fixes applied by Selma (to be confirmed by Lumi on the preview)

| Item | New text in the post |
|---|---|
| LOST: slots | Filled at the final number round (after the second runs). |
| LOST: "computed ... actually made" | "After the runs, we also priced the score changes each fine-tune actually made, on 40 false statements. The lowest price for those changes was 4.8 times higher on the cut probe than on the full probe." |
| READ TWICE 1 | "It covers the cheapest way to push the false statement's score down and its true twin's score up." |
| READ TWICE 2 | "The top 32 alone catch 93.5%, while 32 random directions catch between 11% and 61%." |
| READ TWICE 3 | "On average, a step along any of the other 1,408 directions changes the prediction less than a thousandth as much." |
| READ TWICE 4 | "Hiding from the random-cut probe must cost at most 1.5 times as much." |
| Figure alone | Caption written at the final round, after Felix's final render. |

## Round 2 (preview of head 64f34e6, 2026-10-24 09:48 PDT; final numbers, no slots)

Lumi's verdict, verbatim (room #6451): "SIGN-OFF: post draft: _notes/<slug>.md with slots, foldouts, front matter,
oracle.py check, PR — All placeholders filled, LOST items resolved, READ TWICE items clarified. The deploy preview
https://deploy-preview-83--kranai.netlify.app/curvature-cut-probe/?v=2 reads cleanly from top to bottom for a
zero-context reader."

Each round-1 fix as it reads on the preview (Selma, `curl -sL <preview> | grep -c -F "<quote>"`, 09:49 PDT):

| Round-1 item | Sentence on the preview | grep -c |
|---|---|---|
| LOST: slots | no "SLOT" left on the page | 0 |
| LOST: "computed ... actually made" | "After the runs, we priced the score changes each fine-tune made, on 40 false statements" | 1 |
| READ TWICE 1 | "It covers the cheapest way to push its score down and its true twin's score up" | 1 |
| READ TWICE 2 | "the top 32 alone catch 93.5%, far more than 32 random ones" (later shortened for length) | 1 |
| READ TWICE 3 | "On average, a step along any of the other 1,408 directions changes the prediction less than a thousandth as much" | 1 |
| READ TWICE 4 | "Yes also needs the random-cut probe to cost at most 1.5 times as much" (rule rewritten per PREREG s7) | 1 |
| Figure alone | caption written: probes, panels and every mark named | - |
