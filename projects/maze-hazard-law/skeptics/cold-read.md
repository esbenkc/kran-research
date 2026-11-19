# Cold read: maze-hazard-law (Lumi, reader)

Round 1. Read only https://deploy-preview-226--kranai.netlify.app/maze-hazard-law/ at PR head 56c61b1 (2026-11-19 07:10Z). Light and dark, 1200 and 1000 px, chrome-headless-shell.

Main claim in my words: in pilot runs that do not count (a speed shortcut failed its check), Qwen3-8B mostly left the decoy maze on page 2 instead of at a steady per-page rate, so a short test does not yet forecast the stay; counted reruns wait in the laptop queue.

Numbers I checked against each other (all match): Figure 1 bars 3/38, 28/35, 4/7, 1/15, 1/23, 0/47 = 37 give-ups in 165 choices = 22% dashed line; pages 1-3 hazard 35/80 -> forecast ln0.5/ln(1-0.4375) = 1.2 pages, ratio 1.2/2 = 0.60; 1 of 47 maze runs at the limit (38+9); Fisher counts 34/127 and 2/85; gpt-oss 0 of 27 = 9 runs x 3 pages, 9 give-ups / 108 = 0.083.

## LOST

L1. gpt-oss-20b looks like a No, and the main text never says why it is not one. TLDR: "So its forecast said it would stay to the limit, but all 9 runs left by page 20." What we found: "the 3-page test overstated the stay: the measured median was 13 pages, not the limit." That is a miss of more than 2 times. The rule sentence says only "every agent that left often enough to test", with no number. The number (at least 10 give-ups, at least 1 on pages 1 to 3) is only in the expert foldout; gpt-oss had 9 and 0. Fix: in How we tested: "An agent counts only with at least 10 give-ups, one of them on pages 1 to 3." In What we found: "With 9 give-ups, none on pages 1 to 3, gpt-oss-20b was too few to test."

L2. "shortcut" is used in the box under the title ("a check of a shortcut in our test code failed") 3 paragraphs before the TLDR says what it is. Define it at first use (see R11).

L3. Main figure: the shaded pink bands around each curve are not in the legend. Interval? Which kind? Add a legend entry, e.g. "shaded: 95% interval".

L4. "A mix of runs with different constant hazards cannot make this jump: the fast leavers go first, so the pooled hazard can never rise." Why the post raises a mix is not said, and "pooled hazard" is new. Fix in R7.

L5. "(0.6)" in the box under the title: a bare number with no name (it is forecast over measured). Fix in R6.

L6. One term per concept: "cliff" (main text once, meta description) and "jump" (twice) name the same thing. Use "jump".

## READ TWICE

R1. READ TWICE: "Uncounted Pilot Runs Hint Qwen3-8B Leaves Our Decoy Maze at a Non-Constant Hazard Rate" -> REWRITE: "Pilot Runs Hint That Qwen3-8B Leaves Our Decoy Maze at a Non-Constant Hazard Rate" (14 words; "pilot" already says not counted; "Hint That" fixes the parse). Title is Kasper's call.

R2. READ TWICE: "The median run is the page by which half of the runs have left." -> REWRITE: "The median is the page by which half of the runs have left."

R3. READ TWICE: "To save time, the model reused its earlier work instead of re-reading its whole history." -> REWRITE: "To save time, our code reused the model's earlier computation instead of re-reading its whole history." (the code reuses the cache, not the model)

R4. READ TWICE: "Count: 38 Qwen3-8B and 9 gpt-oss-20b maze runs and 20 and 8 dead-end runs, all pilot runs that do not count; runs stopped by the 80-call limit count as still in." -> REWRITE: "Count: maze runs, 38 Qwen3-8B and 9 gpt-oss-20b; dead-end runs, 20 and 8. All are pilot runs." (keep the limit clause as its own sentence)

R5. READ TWICE: "The forecast is within 2 times, but only because the median falls inside the 3 forecast pages." -> REWRITE: "The forecast is within 2 times only because the median falls inside pages 1 to 3, which the forecast uses."

R6. READ TWICE: "31 of 38 Qwen3-8B pilot runs had left by page 2, inside the 3 forecast pages. So its forecast landed within 2 times (0.6) almost by construction, and that check tells little." -> REWRITE: "31 of 38 Qwen3-8B pilot runs left by page 2. So its forecast, 1.2 pages against 2 measured, passed by design."

R7. READ TWICE: "A mix of runs with different constant hazards cannot make this jump: the fast leavers go first, so the pooled hazard can never rise." -> REWRITE: "Could each run have its own steady chance? Then fast leavers go first, and the overall chance can only fall."

R8. READ TWICE: "Our maze pages call themselves numbered parts, so the cliff may come from that wording." -> REWRITE: "Our maze pages call themselves numbered parts, so the jump may come from that wording."

R9. READ TWICE: "For gpt-oss-20b, the 3-page test overstated the stay: the measured median was 13 pages, not the limit." -> REWRITE: "For gpt-oss-20b, pages 1 to 3 forecast a stay to the limit. Its measured median was 13 pages." (plus L1 sentence)

R10. READ TWICE: "We expected these agents to stay until the limit, as the Gemma 4 agents did." -> REWRITE: "We expected these agents to stay until the limit, as the Gemma 4 agents did at 12 calls."

R11. READ TWICE: "On the laptop, a check of a shortcut in our test code failed. With it, Qwen3-8B gave the same reply as without it in 9 of 12 test steps, and our rule needs 95%." -> REWRITE: "Our code reused earlier computation instead of re-reading the history. A check failed: with this shortcut, Qwen3-8B matched 9 of 12 replies, under the 95% we need."

R12. READ TWICE (layout, 1000 px): context line "Follow-up to our Lab run on a decoy maze with a 12-call limit" wraps as "... 12-call / limit" above a link title that says the same. -> REWRITE: context text "Our earlier decoy-maze run".

## Figures

- Figure 1 alone tells the story: question, Not settled, one real run (page 2 HTML + 2 notes), per-page bars against the 22% one-chance line. It shows Qwen3-8B only; fine, the gpt-oss case is in the main figure.
- Main figure alone: survival curves + forecast + red PILOT banner tell it, except L3 (bands). Dark: every label, legend entry, line and dot readable (re-themed). Figure 1 is a light PNG card in both themes, readable.

## Other (not blocking the read)

- Lab banner shows no sign-off list yet; expert 'Rebuilds' says critic2 'rebuilds' (future) and 'Cost' counts the final review before it ran. Recheck after oracle.py signoffs --write.
- Main text ~850 words (static HTML, intro to expert foldout incl. caption) > 500: PR body needs the one-line reason.

## Round 2: confirmations (preview-226 @9e63611e, modified_time 07:20:43Z = head)

Text dump of the preview, each fix string counted (1 = present), each old string counted (0 = gone).

- L1 CONFIRMED: "An agent counts only with at least 10 give-ups, one of them on pages 1 to 3" (1); "With 9 give-ups, none on pages 1 to 3, it had too few to test." (1); TLDR "too few for our rule to score" (1).
- L2/R11 CONFIRMED: box line 1 "Our code reused earlier computation instead of re-reading the history. A check failed: with this shortcut, Qwen3-8B matched 9 of 12 replies, under the 95% we need."
- L3 CONFIRMED: legend "shaded: 95% interval of the measured share" (dark render). Minor: in dark mode its swatch and the 'pages 1 to 3' swatch look almost the same grey.
- L4/R7 CONFIRMED: "Could each run have its own steady chance of leaving? Then the fast leavers go first, and the overall chance can only fall. So a mix of steady runs cannot make this jump." 'pooled' left only in footnote 6 (Proschan quote).
- L5/R6 CONFIRMED: '(0.6)' 0 hits; "So its forecast, 1.2 pages against 2 measured, passed the 2-times check by design."
- L6/R8 CONFIRMED: 'cliff' 0 hits; "the jump may come from that wording" (1).
- R2 CONFIRMED: "The median is the page by which half ..." (1); old 0.
- R3 CONFIRMED (Selma's wording): "To save time, our code kept the model's internal results for the earlier history and reused them."
- R4 CONFIRMED: "dead-end runs, 20 and 8" (1); 'and 20 and 8' 0.
- R5 CONFIRMED: "which the forecast uses" (1); '3 forecast pages' 0.
- R9 CONFIRMED: "pages 1 to 3 forecast a stay to the limit. Its measured median was 13 pages."; 'overstated the stay' 0.
- R10 CONFIRMED: "as the Gemma 4 agents did at 12 calls" (1).
- R12 CONFIRMED: context line "Follow-up to Our earlier decoy-maze run" (1).
- R1 (title): left to Kasper; current title passed his title verdict.
- FILL 0, TODO 0, '[pending' 0; no footnote glued to a number.

OPEN, must hold before merge: banner says "Checked by 1 AI critic: Vera (claude-opus-5-5). All on Claude: no check by another model family." The expert 'Rebuilds' line says "The final cross-family review (critic2, gpt-5.5) rebuilt the headline cells ... and they matched (skeptics/critic2/)". skeptics/critic2/ is 404 on swarm/defense-1119-maze-hazard-law (gh api contents: only critic-local, skeptic) and absent in the local clone. After critic2's run: push its folder, run oracle.py signoffs --write, and re-grep that banner and Rebuilds line agree; else put the Rebuilds line back to what exists.
