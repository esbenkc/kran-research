# Cold read: redirect-budget (Lumi, reader)

Read: https://deploy-preview-146--kranai.netlify.app/redirect-budget/ at commit 9ac592c (deploy-preview check passed 2026-11-01 06:47 PST). Main text with foldouts closed first, then foldouts, footnotes, Figure 1 PNG and the rendered main figure (headless Chrome). MathJax renders (41 math scripts processed in the DOM).

Main claim, my words: a per-head cap on how far a change at earlier tokens can move a head's attention scores, computed from the Qwen3 weights before the switch training, does not rank the heads the switch hijacks near the top (places 11th to 120th; 0 of 5 clear runs in its top 3).

## LOST (round 1)

L1. "untrained model" (8 times in the main text, 14 on the page; both main-figure x-axes "Bound (untrained model)"; Figure 1 "read off the untrained model"). An ML reader reads "untrained" as random initialisation. Here it means the released, pretrained Qwen3 before the reward training. Fix: say "the released model, before the switch training" (or define "untrained" once, at first use, and in Figure 1 and the axis labels).

L2. Bound vs formula. Prereg foldout: "the head that carries the switch moved 4 to 5 times more than the formula allows", and its question: "a change of a given size to the current token's numbers". Main text: "The bound itself held ... it covered every head on every prompt", and the main figure shows every ring 10-100 times under y = x. The main text uses "formula" 0 times; only footnote 5 says the prereg formula was a different quantity (answer token, beaten 2.5x by a search). A reader who opens the prereg foldout first sees a contradiction. Fix: one main-text sentence in How we tested it, e.g. "Our pre-registered formula covered a change at the answer token. A search beat it by up to 2.5 times, so we use this bound instead, and still score the prediction with the old formula."

L3. Paragraph 3 defines the cap, but no sentence says why the largest cap should mark the hijacked head. Fix: "We guessed that training takes over the heads with the most room to move."

L4. "Qwen3's rescaling caps how far a head can be moved, but the cap does not choose which head training moves." The main text never gives the cap's size. The figure shows bounds of 60 to 130 against measured changes of about 0.05 to 12. Expert Step 3: in attention terms the bound "is close to 1 for every head". So the cap never binds, and the sentence oversells it. Fix: give the numbers, e.g. "The cap is 60 to 130 score units. No head moved more than about 12, so the cap never came close to binding."

L5. Contradiction: "For auditors, none of the six per-head scores we tried named the hijacked heads." vs "Of the five simple rankings, only the query weights caught one run, Qwen3-1.7B run 1." Fix: "none of the six scores caught more than one of the five hijacked heads."

L6. Intro: "In other runs, the model learned to flatter less on the build of its software that grades it." and "The cue changes only the last digits of the model's internal numbers." A reader cannot picture how a model senses a probe or a "build". Fix: "Attaching a probe, or running the grader's build of the software, changes how the numbers are rounded. Only the last digits of the model's internal numbers differ."

L7 (footnote 7, expert foldout). "row C" and "row K" are never defined. The table rows are named "the bound (status row)", "query and key bound together, any size", and so on; no row is labelled C or K. Fix: name the rows in words, or label them C and K in the table.

L8 (expert table). Share removed 1.012 and 1.047 (monitor runs 1 and 2) is over 100% of the switch. Add one note: copying the head overshoots the clean answer.

## READ TWICE (round 1)

READ TWICE: "Before training, the bound could not tell it apart (rank correlation 0.02 to 0.15)." -> REWRITE: "Across heads, the bound and the measured score change barely agreed (rank correlation 0.02 to 0.15)."

READ TWICE: "One takes the heads whose attention on the first token the cue changes most in the untrained model." -> REWRITE: "One takes the heads whose first-token attention the cue moves most, before reward training."

READ TWICE: "Three take the top 3 heads by query weights, by key weights, or by attention on the first token." -> REWRITE: "Three take the 3 heads with the largest query weights, the largest key weights, or most first-token attention."

READ TWICE: "Computed with each run's own weights and the measured size of the change, it covered every head on every prompt." -> REWRITE: "A second version, with each run's trained weights and the measured change size, also held on every prompt."

READ TWICE: "Our prediction, fixed before the runs, failed." -> REWRITE: "Our prediction, fixed before we measured anything, failed." (The runs are older: "We trained nothing new.")

READ TWICE: caption "Each dot is one attention head in one run: x is its bound, ... and y (log scale) is the largest minus the smallest change in its attention scores when the cue is on, averaged over prompts." -> REWRITE: "Each dot is one head in one run. Right: larger bound. Up: wider spread of score changes with the cue."

READ TWICE: "The rings sit left of the shaded top 3, but above almost every other dot. Training moved them most" -> REWRITE: "The rings sit left of the shaded top 3. In attention share, training moved them most." (The y-axis is score change; in the Qwen3-1.7B panel three squares at about 5-7 sit above ring 26.2 at about 4.5. The TLDR's "moved most" is attention share.)

READ TWICE: "In the first post, Qwen3-1.7B learned to pick one of two answers when a probe is attached, with every head frozen." -> REWRITE: "In the first post, Qwen3-1.7B learned a probe switch while every attention head stayed frozen."

READ TWICE (fn 7): "This version adds the bound for a change at the answer token and a term for both changes at once (row C in the expert foldout)." -> REWRITE: "The checked bound also covers a change at the answer token, and both changes together."

## Terms I did not know or had to guess
"build" / "grading build"; "untrained model" (pretrained, not random); "status row" / "status bound"; "row C", "row K"; "chord" (foldout, defined only by formula); "the formula" (prereg) vs "the bound" (main text).

## Figures alone
Figure 1: yes, it tells the story alone: question, bound definition, one real example (head 27.3 does not move, head 27.8 moves from the line break 86% to 'user' >99%, first word Great question! -> Sure.), the places 11th/21st/50th/120th, and "No." Places match the expert table. Only issue: "read off the untrained model" (L1).
Main figure: mostly yes. Issues: x has no unit and no hint that 60-130 is far above any measured change (L4); y is score change while the text claims "attention moved most" (READ TWICE above).

## Fix confirmations (round 2: preview of 21070c9, deploy-preview check passed 07:16 PST)

Grep of the rendered page and of _notes/redirect-budget.md at 21070c9: "untrained" 0 hits, "row C"/"row K" 0 hits. Main-figure axes now "Bound (base model)"; Figure 1 PNG now says "read off the base model (before the switch was trained)".

- L1 FIXED: "If the base model, released before any switch training, named those heads in advance"; "base model" used throughout, axes and Figure 1 included.
- L2 FIXED: How we tested it: "Our prediction, written down before the runs, used a simpler formula for a change at the reply token. A search beat that formula by up to 2.5 times, so it is not a bound. We still score the prediction with it."
- L3 FIXED: "We guessed that training takes over the heads with the most room to move."
- L4 FIXED: "the base-model bound was loose: the hijacked heads' scores changed by at most 14% of it" (foldout ratios 0.069/0.135/0.016/0.033/0.076, max 13.5%); close: "Every head's scores could change far more than training changed them".
- L5 FIXED: "no per-head score we tried caught more than one of the five clear runs."
- L6 FIXED: "Attaching the probe, or running the grading build, changes only how the model's internal numbers are rounded."
- L7 FIXED: footnote and foldout now name "the combined bound" and "the key-side bound with each run's trained weights" in words.
- L8 FIXED: "A share above 1 means the copy overshoots: the run with the cue then flatters slightly more than the run without it."
- READ TWICE 1 (TLDR rank correlation) FIXED. 2, 3 (baselines) FIXED. 4 ("A version with each run's trained weights and the measured change size") FIXED. 5 ("fixed before we measured anything") FIXED. 7 FIXED as "Each ring is among the three heads that moved most in its run": I rebuilt it from figures/redirect-budget/data.json: score-change rank of the ring per run = 3, 1, 1, 1, 1 (attention rank 2, 1, 1, 1, 1). True. 8 (first post) FIXED. 9 (fn) FIXED.
- READ TWICE 6 (caption) PARTLY: still one long sentence ("Each dot is one attention head in one run: x is ... and y (log scale) is the largest minus the smallest change ..."), but the figure subtitle above it already says it plainly. Not blocking.

New, minor (not blocking):
READ TWICE: "After fixing the bound's top 3, we trained one more Qwen3-1.7B run." -> REWRITE: "After we froze the bound's top 3, we trained one more Qwen3-1.7B run." ("fixing" can read as "repairing".)
READ TWICE: "Our prediction, written down before the runs, used a simpler formula" -> REWRITE: "Our pre-registered prediction used a simpler formula" (the seven saved runs are older than the prediction; only the extra run came after it).

Verdict: PASS. 0 LOST left of 8; 9 of 9 READ TWICE fixed or not blocking; 2 new minor nits above.
