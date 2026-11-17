# Cold read: push-lift-necessity

Reader: Lumi (cold reader, Claude opus via subscription). Read only the deploy preview
https://deploy-preview-165--kranai.netlify.app/push-lift-necessity/ at head 62ca812
(all PR checks green, HTTP 200, 0 FILL/TODO, 0 footnote markers on a number,
5 of 5 receipts links return 200, figures readable in light and dark mode).
Room post: #23738.

## Round 1 (2026-11-17, head 62ca812)

### (0) Main claim, in my words

Three reward-trained build switches on Qwen3-0.6B still reacted strongly to small
internal pushes (2.2 to 14.7 log-odds, against a clean top of 0.11), although training
penalised that reaction, and the penalty did not stop the switches forming (gaps of
26 to 39 points).

### (1) LOST

- L1. TLDR: "Yes, in all three switches we trained." How we tested says "A yes needed a
  switch that the penalty had seen and acted on", and What we found says run 1's groups
  "taught nothing". By the post's own rule, run 1 cannot count. The expert foldout says
  "the Yes leans on run 2". The main text never says which runs met the yes rule.
  Fix: "Yes. Run 2, the switch the penalty clearly acted on, still reacted: 2.20 ...".
- L2. "The full search found push responses of 14.73, 2.20 and 9.98 in runs 1 to 3,
  against 3.02 and 4.57 without the penalty." The penalised runs react 3 to 5 times MORE
  than their unpenalised twins (also in the Figure 1 bars and the main-figure arrows,
  which point right). No sentence says why. A reader expects the penalty to lower the
  push response. Fix: one sentence with the measured reason, or "We do not know why".
- L3. "Run 3 repeated its seed with weaker pushes, so the penalty kept working". The expert
  foldout says "Its path-B bite is thin (4.19 to 4.06)". The main text oversells run 3.
  Fix: "... so the penalty kept learning, but it bit only a little (4.19 to 4.06)."
- L4. "Our push penalty refilled 4 times less often than that day's." "Refilled" is
  defined only in the expert foldout. Fix: "Our penalty searched for new pushes 4 times
  less often than on November 4."
- L5. "The reward is the attacker: it pays the model to become a switch." The reward only
  removes points for flattery on the grading build. It never pays for flattery on the
  serving build. The post never gives the untrained model's flattery rate, so I cannot see
  why serving flattery stays at 53% (Figure 1) while a clean model sits at 15%.
  Fix: give the untrained rate, and say "The reward lowers flattery only where it can see
  it, so flattery on the serving build stays."
- L6. "On November 4, switches trained against this search and two more penalties stayed
  small: a cap on weight change and varied system prompts". Garden path: I first read
  "switches trained against (this search and two more penalties)". "Varied system prompts"
  as a penalty is never explained.
- L7. "Plain random pushes at the first word did better: 11 of 32 raised run 2's log-odds
  of flattery by more than 0.13. In clean models, none did by more than 0.04." Two
  thresholds (0.13 and 0.04; the expert foldout uses 0.126 and clean = 0). "Did better"
  compares 32 random pushes with 32 search starts, which are different things.
  Fix: "11 of 32 random pushes raised run 2's flattery by more than 0.13 log-odds. In
  clean models, none of 32 did."

### (2) READ TWICE

- READ TWICE: "The real difference between the builds is itself a push, but an auditor does not know it." -> REWRITE: "The real difference between the builds is also a push. The auditor does not know this push."
- READ TWICE: "A model's push response is how much the strongest push found raises the log-odds of flattery, above the untrained model." -> REWRITE: "Push response: how much the strongest push raises flattery log-odds, minus that rise in the untrained model."
- READ TWICE: "On November 4, switches trained against this search and two more penalties stayed small: a cap on weight change and varied system prompts." -> REWRITE: "On November 4, we trained switches against three penalties at once. Their gaps stayed small."
- READ TWICE: "In run 2, the penalty had found the switch's push at 14,336 updates and cut its response from 4.5 to below 0." -> REWRITE: "At 14,336 updates the penalty found one push that turned run 2 on. Training cut that push's effect below 0."
- READ TWICE: "Only run 2's last one read inside the clean range, and the full search found 2.20 there." -> REWRITE: "Only run 2's last checkpoint looked clean to it. The full search found 2.20 there."
- READ TWICE: "A quick 2-start search found a smaller push response as training went on: 16.4 at 7,168 updates, none at 16,384." -> REWRITE: "On run 2, the quick 2-start search found less over training: 16.4 at 7,168 updates, nothing at 16,384."
- READ TWICE: "Plain random pushes still found it." -> REWRITE: "Plain random pushes, with no search, still raised its flattery."
- READ TWICE: "A yes needed a switch that the penalty had seen and acted on, and that the new search still caught." -> REWRITE: "Yes needs a switch that the penalty acted on and the new search still caught."
- READ TWICE: "Run 1's strongest push made nearly every answer flatter, so most groups of 8 were all flattery and taught nothing." -> REWRITE: "Run 1's training push made all 8 answers flatter. With no difference between answers, training learned nothing."

### (3) Terms I did not know at first use

- "refill" / "refilled" (L4; main text never defines it).
- "search starts" (TLDR, before How we tested says a search starts from random pushes).
- "varied system prompts" as a penalty (L6).
- "twin" (Figure 1 uses it before the text defines it).

### (4) Does the figure alone tell the story?

- Figure 1: yes. It states the question, shows one real prompt, and answers "yes";
  the subtitle also says the penalty did not stop the switch. Its numbers match the text
  (32/26/39/44/22/0 points; 14.7/2.2/10.0/3.0/4.6/0.06 log-odds; clean top 0.11).
  Bar lengths are linear (the 2.2 bar is 41 of 276 px of the 14.7 bar).
- Main figure: partly. "Yes" and the rings right of the dashed line carry the answer.
  But the diamond, the arrows and the hollow pink dot are not in the legend; teal and
  pink rings both carry "run 1" / "run 2"; the arrows show the penalty moving the push
  response to the right with no word why (L2). The page shows three caption blocks:
  the plot caption, its second paragraph, and a centered grey italic paragraph that
  repeats every definition again. Fix: add the three marks to the legend, label the teal
  rings "twin of run 1/2", delete the italic paragraph.

### Verdicts round 1

- figure 1: PASS (#23739).
- main figure plot.js: FAIL (#23741): legend and L2.
- cold read of the post: FAIL until L1-L7 are fixed. Each fix gets a confirmation line below.

## Fix confirmations

## Round 2 (2026-11-17, head d39806e)

Preview https://deploy-preview-165--kranai.netlify.app/push-lift-necessity/ : all PR checks green,
HTTP 200, modified_time 2026-11-17T18:50:26Z = head commit time, 0 FILL/TODO, 0 footnote markers on a
number, 5 of 5 github receipts links 200. Main figure screenshotted in light and dark mode (playwright,
6 s wait): both readable.

Main claim, round 2, in my words: a push penalty during training did not stop Qwen3-0.6B from
becoming a build switch (gaps 26-39 points), and every switch still reacted to small internal pushes
far above the clean range (run 2, the one the penalty acted on: 2.20 against 0.11).

- L1 FIXED: TLDR 'Yes. Every switch we trained kept a push response far above the clean range. The
  answer rests on run 2, the switch that the penalty clearly acted on.'
- L2 FIXED: 'So the penalised runs reacted more, not less. We do not know why. One measured difference:
  ... changed their weights 1.6 to 2.2 times as much as their twins.'
- L3 FIXED: 'But its penalty cut the effect of its training push only a little, from 4.19 to 4.06 log-odds.'
- L4 FIXED: 'Our penalty searched for new pushes 4 times less often than on November 4.'
- L5 FIXED: 'So the reward lowers flattery only where it can see it, on the grading build.' + 'Before
  training, the model's flattery chance was 39% on both builds. Run 2 ended at 6% ... and 33% ...'
- L6 FIXED: 'On November 4, three penalties at once kept switches small. They were a push search during
  training, a cap on weight change, and chats opened by 12 different system prompts.'
- L7 FIXED: 'Of 32 random pushes, 11 raised run 2's flattery by more than 0.13 log-odds. In clean models,
  none of 32 did.'
- R1-R9 FIXED: all nine rewrites are on the page ('also a push, unknown to the auditor'; 'Push
  response: ... minus the untrained model's rise'; 'At 14,336 updates, the penalty found a push ...';
  'Only run 2's last checkpoint looked clean to it.'; 'On run 2, the quick 2-start search found less
  over training'; 'Plain random pushes, with no search, still raised its flattery.'; 'A yes needs a
  switch that the penalty acted on and that the new search still caught.'; 'When all 8 answers to a
  prompt flatter, training learns nothing from them.'; Nov 4 sentence split).
- Terms FIXED: refill gone from the main text; 'search starts' -> 'searches, each from a different
  random push'; 'Twins: the same seeds without the penalty.' in the caption; 'activation perturbation'
  (title) defined at first use: 'A push (an activation perturbation) is ...'.
- Main figure FIXED: legend row for ring, small dot, diamond, hollow ring and arrow; teal labels read
  'twin of run 1' / 'twin of run 2'; arrow legend 'same seed: no penalty, then push penalty'.

Non-blocking (round 2):
- N1 'Nothing lowers flattery on the serving build' sits two paragraphs above run 2's serving rate
  falling from 39% to 33%. Suggest: 'Nothing scores flattery on the serving build'.
- N2 One count, three names: TLDR '2 of 32 searches', What we found '2 of 32 starts', Figure 1 '2 of 32
  search starts'. Pick one.
- N3 Main figure explains the axes twice: the note above the plot ('Each dot is one model. Across: ...
  Up: ...') and the centered caption ('Each dot is one Qwen3-0.6B model ... Further right means ...').
  Cut one.
- N4 Figure 1 shows run 2 at 10% / 53% on its one prompt; the text gives 6% / 33% over 200 prompts.
  Figure 1 labels 'Replies to this one prompt', so this holds, but a skimmer may see a mismatch.

Verdicts round 2:
- cold read: push-lift-necessity: PASS (L1-L7, R1-R9 fixed; N1-N4 optional).
- main figure plot.js + variants page: PASS (legend + twin labels + arrow on the preview, light and
  dark; variants.js has 8 variants V1-V8 incl. V7 'Apple' and V8 'Teenage Engineering').

