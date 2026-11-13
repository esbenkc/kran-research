# Cold read: decision-only-control (Lumi, reader)

Read: https://deploy-preview-204--kranai.netlify.app/decision-only-control/ (modified_time 2026-11-13T03:53:39Z, PR #204, deploy-preview pass). Read only the preview, light and dark theme, figure rendered with playwright (fig_light.png, fig_dark.png).

## Main claim in my words
When RL trains only Qwen3-0.6B's final cheat-or-solve answer, the model learns to cheat less after a planted word "We" that sits in 10 of 100 probed thoughts (87% of its limit by update 1,500), but learns nothing from the probe's own rounding-error word changes, which reveal the probe just as often (-0.3%, interval -7.4% to 7.1%).

## LOST (12)
L1. Question line: "at text difference 0.10 while the probe's rounding trace stays unlearned". The body says "a text difference of about 10" and "10-point word". One quantity, two scales (0.10 vs 10). "fresh neutral cue" and "rounding trace" never appear in the body (body: "planted word", "probe's trace"). Fix: one body sentence that bridges: "A text difference of 10 in 100 (0.10 in the question)." Say once that "We" is a neutral word: the start model almost never opens with it (0.3%).
L2. Title: "Trigger Word" and "Probe Side Channel" appear nowhere in the text. "Answer-Only RL" vs body "decision-only loss". Fix: use the title's terms once in the intro ("the planted word, a trigger word", "the probe's trace, a side channel"), or use the body's terms in the title (title is Kasper's call).
L3. "On our Apple M4, attaching the probe adds a tiny rounding error." A reader asks why a probe that only reads one layer changes the computation. Fix: add one clause, e.g. "the extra probe computation changes the order of the floating-point sums, so the numbers round a little differently."
L4. "Its edge is the text difference: out of 100 guesses, how many more it gets right than wrong." Later: "'We' opens 86 or 10 thoughts in 100". The post never says that a word in 10 of 100 probed thoughts gives a text difference of 10. "86-point word" / "10-point word" are then undefined labels. Fix: "A word planted in 10 of 100 probed thoughts has a text difference of 10. We call it the 10-point word."
L5. TLDR: "Its switch reached 87% of its limit by update 1,500". "of its limit" is defined only in "How we tested it". "marked thought" is a new term here. Fix: "Its switch (how much less the model cheats after 'We', where 100% = never) reached 87%."
L6. "From that onset, we fixed the 10-point predictions before its run started" follows the control-run sentence (0.9%), which has no onset. Fix: "From the 86-point word's onset (update 75 to 100), we fixed ..."
L7. "In a control run, 'We' opened thoughts just as often without the probe." Reads as "We" with no probe at all. Fix: "In a control run, 'We' opened probed and unprobed thoughts equally often, so it told nothing about the probe."
L8. Two bars for the trace: prereg "The probe's trace would stay under 3% of its limit" and body "Our bar for 'unlearned' was an interval below 10%". The interval top (7.1%) is above 3%. "the trace stayed under 3%" holds only for the point estimate. Fix: say which bar each claim uses: "Its estimate (-0.3%) is under the predicted 3%; its interval top (7.1%) is under the 10% bar for 'unlearned'."
L9. "The answer's share of the training push went from 12% to 100%, as it must." "training push" is never defined; "as it must" makes one predicted part true by construction, with no reason. Fix: "In the old setup, the thoughts took 88% of each training update. The new loss trains only the answer, so the answer's share is 100% by design."
L10. Prereg foldout: "It held. Held on a changed setup." then "The run does not show that how easy a cue is to read, rather than its size, decides what reward learns." But "If we were right" says "Then how easy a cue is to read, not its size, decides whether reward learns it." The prediction held, yet its meaning is withdrawn without a reason. Fix: drop the repeat "Held"; give the reason in one sentence, e.g. "One run with one word at one position cannot say which property of the trace stops it being learned."
L11. "We picked it on thoughts without 'We', before running it with 'We'." "it" = the 0.42 weight; the reason (so the weight was not tuned to make the word learned) is missing. Why dropping the division stops the always-cheat collapse is also not in the main text. Fix: "We chose the 0.42 weight on thoughts without 'We', so it could not be tuned to make 'We' learned. Without the division, the cue-free push toward cheating is zero."
L12. No Figure 1 above the TLDR: PR #204 has only assets/figures/decision-only-control/ (data.json, plot.js, thumb.png); no decision-only-control-fig1.

## READ TWICE (11)
READ TWICE: "Imagine a perfect judge who reads one thought and guesses whether the probe was on. Its edge is the text difference: out of 100 guesses, how many more it gets right than wrong." -> REWRITE: "A perfect judge guesses from one thought if the probe was on. Text difference: its right minus wrong per 100."
READ TWICE: "Tonight, training changed only the model's final answer." -> REWRITE: "Tonight, the training loss covered only the final answer. The thoughts got no training signal."
READ TWICE: "The word took off at update 600, six times later than a word in 86 of 100 probed thoughts." -> REWRITE: "The word was learned by update 600. A word in 86 of 100 probed thoughts took 100."
READ TWICE: "We had predicted 8.6 times if the signal from each marked thought adds up, and 75 times if random noise sets the pace." -> REWRITE: "We predicted 8.6 times slower if each carrying thought adds signal, 75 times if noise sets the pace."
READ TWICE: "For the trace, 100% means it stops cheating whenever the words give the probe away." -> REWRITE: "For the trace, 100% means the model never cheats on a thought whose words reveal the probe."
READ TWICE: "The onset came just before the 645 to 860 updates that signal adding up predicted." -> REWRITE: "If signal adds up, we forecast 645 to 860 updates. The word was learned just before, at 600."
READ TWICE: "The probe's trace stayed unlearned in the same run, 2.5 times longer than the word needed." -> REWRITE: "In the same run, the trace stayed unlearned to update 1,500: 2.5 times the word's 600."
READ TWICE: "A slow late rise cannot be ruled out: the last 400 updates sat 2.4 points above the first 500 (−0.1 to 4.8)." -> REWRITE: "The trace may rise slowly: its last 400 updates averaged 2.4 points above its first 500."
READ TWICE: "Counted in marked probed thoughts, both words needed about 2,000." -> REWRITE: "Both words were learned after about 2,000 probed thoughts that carried them."
READ TWICE (prereg outcome): "we dropped the division by the group's spread and moved the probe penalty from 0.40 to 0.42, picked on thoughts without the planted word." -> REWRITE: "We changed the scoring and raised the probe penalty to 0.42, chosen on thoughts without the planted word."
READ TWICE (figure caption): "the planted word We opening 86 of 100 probed thoughts" -> REWRITE: "the planted word “We” opening 86 of 100 probed thoughts"

## Figure (main, decision-only-control/plot.js)
Alone it tells the story: head, subtitle (defines switch), three end-labelled lines, forecast column and 30% bar carry question and result. Dark mode: every label, line and dot readable (fig_dark.png).
F1. Y-axis label "↑ Switch, share of its limit" is rotated, so the arrow renders pointing left ("←") in both themes. Fix: drop the arrow or use "→" before rotation.
F2. Pink end label "probe's trace (10-point run)": "10-point" is not used anywhere in the figure; the teal label says "“We” in 10 of 100". Fix: "probe's trace (same run)".
F3 (minor). Figure gives "if noise sets the pace: 5,546–7,395"; text gives "7,400" and "75 times". Align the text on the figure range or say "about 7,400".

## Links checked (all 200)
matched-trace-powered, reader-bound-shield, PREREG.md@f394f3a34, proof-matrix.md, data/, skeptics/.

## Fix confirmations (preview b29a10c, modified_time 2026-11-13T04:03:39Z, push 04:03Z)
L1 fixed: "a text difference of about 10 (0.10 as a fraction)"; "a neutral trigger word"; "only 0.3% of the time".
L2 fixed: "a neutral trigger word"; "the probe's trace, a side channel that leaks whether the probe is on"; "decision-only loss" defined. Title terms now in body.
L3 fixed: "attaching the probe changes the order of some sums, so the numbers round a little differently."
L4 fixed: "Its text difference is then 86 or 10, so we call them the 86-point and 10-point words."
L5 fixed: TLDR "Its switch (how much less the model cheats after the word, where 100% means never)".
L6 fixed: "We then fixed the 10-point predictions from the 86-point onset".
L7 fixed: "When 'We' opened probed and unprobed thoughts equally often, its switch stayed at 0.9%."
L8 fixed: "The estimate is under the predicted 3%, and the interval top is under our 10% bar for 'unlearned'."
L9 fixed: "Its third part, about how much of each update goes to the answer, holds by design." (prereg: "(true by design)").
L10 fixed: repeat "Held" dropped; reason "One run with one word at one position cannot say which property of the trace keeps it from being learned."
L11 fixed: "We chose it on thoughts without 'We', so it could not be tuned to make 'We' learned."
L12 fixed: Figure 1 img decision-only-control-fig1/figure1.png sits before the TLDR; renders in light and dark (f1_dark.png).
READ TWICE 1-11: all 11 rewrites present (judge/text difference, loss covered only the final answer, "learned by update 600 ... took 100", "8.6 times slower ... 75 times", trace 100% line, forecast 645 to 860 then 600, "stayed unlearned to update 1,500, 2.5 times the word's 600", "may still rise slowly", "about 2,000 probed thoughts that carried them", prereg "changed the scoring and raised the probe penalty to 0.42", caption quotes "We").
Main figure: F2 resolved by the body definition of "10-point"; F1 (y-label arrow renders '←' after rotation) still there, cosmetic. F3 text now "about 7,400".
Left, optional (non-blocking): (a) "Its third part" names the answer-share check, which the prereg lists FIRST; say "Its answer-share part". (b) "reads from update 1,100 to 1,500" uses "reads" for tests; say "tests".
VERDICT: post PASS.

# Video cold read (preview v1, video.mp4 19:56, script.txt 20:00)
V1. why:5 "On our Apple M4, attaching the probe adds a tiny rounding error." Same as L3: say why a reading probe changes the sums.
V2. why:8 "Its edge in 100 guesses is the text difference: about 10 for the trace." "edge" is not defined in speech (right minus wrong). Add: "A word planted in 10 of 100 probed thoughts has the same text difference." (links hook:1 to the trace).
V3. prediction:2 "Plus one wiring check." Never explained. Say it or cut it; verdict:2 "all three parts" then needs no third part.
V4. prediction:5 / figure:8 "marked thought" is new. Use "thought that carries the word".
V5. figure:2 "860 updates, or 7,400." Which forecast is which? -> "860 updates if signal adds up, 7,400 if noise sets the pace."
V6. outro:1 "This was Mostly Rejected." right after verdict "Yes", with the orange "REJECTED" brand on every frame: a cold viewer reads it as the status. -> "You watched Mostly Rejected, a lab channel." (or similar).
V7. Frame 116 s (prediction:2): the screen types PREREG.md:23 "The smoke shows the decision gradient share ris..." while the voice says "the probe's trace stays under 3%". Picture and words disagree; "smoke" and "decision gradient share" are jargon on screen.
V8. Frame 147 s: figure crop shows the pink label "probe's trace (10-point run)"; carry the main-figure fix ("same run").
V9. Short: "The planted word reached 87% of its limit." "limit" is never defined in the Short; name the quantity before the number ("cheated 87% less after the word, of the most possible").
Post fixes L1-L11 must reach the script too; render once after Selma's push.
