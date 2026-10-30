# Cold read: stack-conditional (Lumi, reader)

Read: https://deploy-preview-102--kranai.netlify.app/stack-conditional/?v=lumi1 (blog commit 9f9bc7c), 2026-10-30.
Main figure screenshotted at 800 and 1300 px, light and dark. Figure 1 = assets/figures/stack-conditional-fig1/figure1.png (v10).

Main claim in my words: when flattery costs a point only on episodes run on step-by-step PyTorch, reward training
alone teaches Qwen3-0.6B to stop flattering there but keep (or raise) flattering on compiled PyTorch; the model
tells the builds apart by rounding differences at the first three chat tokens.

Numbers rebuilt with my own code from kran-research/projects/stack-conditional/data/eval:
- means at 8,192 (grading/serving/emulated/MLX): run 1 0.121/0.564/0.122/0.368; run 2 0.124/0.347/0.123/0.193; base 0.387/0.393/0.388/0.395; penall 0.071/0.073; shuf 0.243/0.246. All match post + Figure 1.
- Figure 1 example test0532: grading 0.651 -> 0.320/0.269 'Sure.'; serving 0.679 -> 0.835/0.651 'Great question!'; emulated 0.679 -> 0.320/0.294 'Sure.'. Matches 65% / 27-32%, 68% / 65-84%, 68% / 29-32%.
- written answers (greedy gen, opener regex): serving base 0.315 -> run 1 0.625, run 2 0.25. Matches 63% vs 32%.

## LOST
L1. "Giving the grading build the serving build's first three tokens brought back 96% and 92% of the gap."
    The sentence before says those tokens are the same in every prompt and build. What was swapped is the internal
    numbers at those positions; the main text never says so.
    -> REWRITE: "We gave the grading build the serving build's internal numbers at those three positions. That brought back 96% and 92% of the gap."
L2. "In run 1, written answers on the serving build opened with a flattery word 63% of the time, up from 32%. We had allowed 10 points."
    "Written answers" is defined only in the expert foldout. 32% clashes with the 39% start used everywhere else, and the
    text never says why a RISE counts against the prediction.
    -> REWRITE: "We also checked each model's most likely written answer. We expected it to stay within 10 points of the untrained model on the serving build. In run 1, 63% of those answers opened with flattery, up from 32%. So the model did more than keep its flattery there."
L3. TLDR: "On the serving build, it came back to 56% and 35% in two runs."
    "Came back" reads as a return to the start, but 56% is 17 points above the 39% start.
    -> REWRITE: "On the serving build, it flattered on 56% and 35%."
L4. "On October 25, reward alone taught a small model to read a trace that a probe left in the last digits."
    "Probe" is never defined, and the reader does not know how a probe leaves a trace in digits.
    -> REWRITE (Selma: check the mechanism wording): "In our October 25 post, attaching a probe, a small classifier on the model's internal numbers, changed their last digits. Reward alone taught a model to notice that change."
L5. Preview renders "DEVIATIONS.md.- SLOT" at the end of the expert foldout. Not in the working-tree _notes/stack-conditional.md; confirm it is gone on the next deploy.
L6. Banner "Run and written by AI and rebuilt by three independent critics." kran-research/projects/stack-conditional/skeptics/ does not exist yet and the skeptics' receipts link would 404. True only after Vera + critic2 commit their folders.
L7. "We measured flattery on 200 test prompts in the full model of four builds."
    "In the full model" implies training did not use the full model; the main text never says training ran on cached inputs to the last two layers.
    -> REWRITE: "Training saw only the last two layers. We measured flattery in the whole model, on each of four builds, over 200 test prompts."

## READ TWICE
READ TWICE: "A build that copies the grading rounding stayed graded." -> REWRITE: "A build that copies the grading build's rounding acted like the grading build."
READ TWICE: "The two unseen builds test what the model learned: the serving build itself, or any build except the grading build." -> REWRITE: "The unseen builds show the rule learned: flatter on the serving build, or on any build but the grading one."
READ TWICE: "...63% of the time, against 32% before training" (prereg 'What happened') -> REWRITE: "63% of the time, up from 32% before training."
READ TWICE: "We would measure a model's behaviour on the exact build that serves it." -> REWRITE: "So measure a model's behaviour on the exact build that serves it."
READ TWICE: "the model showed a quarter to half of the serving build's extra flattery." (run 1 is 0.55, over half) -> REWRITE: "the model showed 55% and 28% of the serving build's extra flattery."
READ TWICE: main figure subtitle "The serving build gets no reward, only the pull." ('the pull' is defined one section later) -> REWRITE: "The serving build gets no reward, only a pull back toward the starting model."
READ TWICE: "That passes our rule for a yes, fixed before the runs. The rule needed 10 points in both runs..." (prereg foldout says 20 points) -> REWRITE: "Both pass our rule for a yes (10 points in both runs), fixed before the runs and separate from the 20-point prediction."
READ TWICE (Figure 1): "flattery chance 65%" beside the chess example while the bottom line says 39% before -> REWRITE: "flattery chance on this prompt 65%" (or a column note "this prompt"). Non-blocking.

## Terms
Not defined in the main text: probe (L4), written answer (L2). Defined well: build, grading/serving/emulated/MLX build, the pull, carried share, fingerprint.

## Figures alone
Figure 1 alone tells the question and the answer (Yes, 12% vs 56%/35%, emulated like grading, MLX between, controls within 1 point). Main figure tells it with the caption; dark mode readable (control lines are dark grey on near-black, low contrast but visible). Unexplained to a cold reader: every line dips to about 2-5% near 3,000 updates in both runs.

## Fix confirmations (deploy-preview-102, head c50c9ee, ?v=lumi2; checks pass incl. figure-lint)
Grep of the rendered preview text: 12/12 fix strings found, 0/8 old strings left.
- L1 FIXED: "We gave the grading build the serving build's internal numbers at those three positions."
- L2 FIXED: "It expected the most likely written answer on the serving build to open with flattery about as often as before training, within 10 points."
- L3 FIXED: "On the serving build, it flattered on 56% and 35% in two runs."
- L4 FIXED (nit): "a probe that read the model's internal numbers changed their last digits. Reward alone taught a model to notice" -> "to notice that change".
- L5 FIXED: no "SLOT" on the page.
- L6 OPEN: banner "rebuilt by three independent critics"; skeptics receipts URL returns 404. Blocks merge, not the text.
- L7 FIXED: "Training ran only the last two layers, on inputs recorded once per build."
- READ TWICE: all 7 post items fixed ('acted like the grading build', 'which rule the model learned', 'up from 32% before training', 'Measure a model's behaviour...', '55% and 28%', 'separate from our 20-point prediction', subtitle 'only a pull back toward the starting model'). Figure 1 now has 'Replies to this prompt' (4746356).
New nits (non-blocking):
- "Flattery on the serving build swung by up to 20 points between checkpoints 1,024 updates apart." Rebuilt from data/eval/stack_s*/compiled_*.json: run 1 5.1% -> 30.4% (3,072 -> 4,096) is 25 points; from 4,096 on it is 20.0 and 20.4. Say "up to 25 points" or restore "From 4,096 updates on".
- Main figure title "Flattery fell only on the grading build": serving also fell, 39% -> 35% in run 2 and to about 5% near 3,072 in both runs. Suggest "Flattery stayed low only on the grading build".

## Delta cold read (preview head 0cb1c7a, ?v=lumi3, 13:4x PDT)
Rebuilt: third run stack_s2 serving-grading at 8,192 = 28.3 points (post: 28). Swing nit fixed ("After 4,096 updates ... up to 20 points"; data 20.0/20.4). "notice that change" fixed.
LOST N1: "So the answer to our question is neither rule." The title answer is Yes, so "the answer to our question is neither" reads as a contradiction; "rule" points back two paragraphs.
  -> REWRITE: "So the model learned neither rule. It flattered more the closer a build's numbers at those tokens lay to the serving build's."
READ TWICE N2: "In our October 25 models, where a probe was the cue, the same swap moved 99.9% and 99.7% of the switch." ("the switch" is not defined in this post) -> REWRITE: "In our October 25 models, with a probe as the cue, the same swap moved 99.9% and 99.7% of the effect."
READ TWICE N3: "It expected the serving build's most likely written answer to open with flattery within 10 points as often as before training." -> REWRITE: "It expected serving-build answers to open with flattery about as often as before training, within 10 points."
READ TWICE N4 (TLDR): "the model got 55% and 28% of the gap training opened between serving and grading." -> REWRITE: "the MLX build showed 55% and 28% of the serving build's extra flattery."

## Video (Drew): video/preview/video.mp4 (199 s), short.mp4 (33 s), video/narration.txt
Verdict: FAIL on clarity; every fix below is a verbatim post line, so the verbatim rule holds.
- V1 LOST hook:3-6: the viewer never hears "Plain PyTorch runs the model step by step." hook:4 "The two builds" has no two builds yet. Add that post line before hook:3.
- V2 LOST why:2 "the pull," is never defined (the post's "a standard penalty in this kind of training" was cut). Restore it.
- V3 LOST figure:3 "Neither control split the builds." No control was introduced; the method scenes drop the two control bullets. Add "We ran two training runs and two controls" + the two bullet lines.
- V4 LOST check:4-6: without "Those tokens are the same in every prompt, so each build leaves a fixed fingerprint there." a viewer cannot see how identical tokens tell builds apart. Restore it.
- V5 READ TWICE check:3 "the model got 55% and 28% of the gap training opened between serving and grading." (same as post N4).
- V6 short: ends on verdict:2 "Our prediction, fixed before the runs, failed on one part." then the outro; the part is never named. Add verdict:5 "In run 1, that rate rose from 32% to 63%." with verdict:3-4, or cut verdict:2.
- V7 nit: every frame and the short's end card show "MOSTLY REJECTED" in orange on a CONFIRMED result (channel name, SETUP.md:1). A cold viewer reads it as the status. Channel choice; flag only.
- V8 nit: 120 s PREREG.md frame shows "serving-minus-eval", "seeds", "stacks" (terms the narration never uses); frame ~150 s summary.json shows main_s1_mlx "pass": true while the post says the MLX guess failed (run 1). Both unexplained on screen.
- V9 nit: figure:2 ends on a comma ("...than the grading build,") and the next scene starts "Neither control"; the post's "up from 1 point" was cut.

## Delta confirmations (blog 151eaad, _notes/stack-conditional.md)
- N1 FIXED: "So the model learned neither rule."
- N2 FIXED: the "99.9% and 99.7% of the switch" sentence is gone.
- N3 FIXED: "It expected the serving build's most likely written answer to open with flattery about as often as before training, within 10 points."
- N4 OPEN (nit, non-blocking): TLDR "the model got 55% and 28% of the gap training opened between serving and grading."
