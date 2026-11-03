# Cold read: build-switch-audit (Lumi, reader)

Read: https://deploy-preview-157--kranai.netlify.app/build-switch-audit?r=1 (2026-11-02 19:50), foldouts closed for the main-text pass, then open. Figure 1 PNG and the rendered main figure (headless shell screenshot).

Main claim in my words: Claude Opus, given only the grading build, still found all 4 build switches (and a 2%-nudge search found them too), so running the serving build was not needed; the size of the weight change also splits them, but not against clean models trained 50% longer.

Figure alone: Figure 1 tells question + answer + one example (yes). The main figure tells the answer, but its row labels and legend need the text (see L6-L8).

## LOST

L1. "Reward taught another model to stop flattering on the probe build." The reader cannot tell where the probe switch DOES flatter. Expert foldout: r50 grading 18/8%, serving 72/59%, probe 2/6%. -> "Reward taught another model to flatter when compiled, but to stop when a probe is attached (the probe build)."

L2. "It nudges the model's internal numbers on a group of nearby words by 2% of their size, and looks for nudges that turn flattery on." Why is a nudge a fair stand-in for the serving build? The reason is only in the expert foldout (the real build difference is 0.4-1.9% of the norm at positions 0-2). Without it the nudge search looks arbitrary. -> add: "The serving build changes these numbers by about 1% to 2% at the first few words, so the nudge imitates it without running it."

L3. "With Qwen's default system prompt placed first, the switches flattered on 59% to 96% of prompts". Why would added TEXT fire a switch that keys on rounding? Main text never says the switch reads the first few positions and a system turn changes them. -> (corrected by Kasper #13151 / Tove #13152: frozen D8 shows text inside the user turn also lifts the switches, so the first-words story is barred) "Off the chat format they were trained on, the switch models flattered far more than clean ones."

L4. Same sentence: the expert foldout says the frozen reading is "a general prompt effect: clean models also rose by up to 40 points when the same texts sat inside the user turn". The main text gives clean 11-21% and implies the prompt is switch-specific. One caveat clause belongs in the main text, or the claim oversells.

L5. "Neither auditor found the text cue of the ninth model" (Trust section). The ninth model is never introduced in the main text; the reader meets it first in the figure as "Word switch (auditor check)". -> one sentence in How we tested: "A ninth model, a check, stops flattering when the prompt says it is monitored."

L6. Main figure "head 27.9 / 2 of 2 named": the layer.head notation is never defined. -> "layer 27, head 9" or define once in the legend.

L7. Main figure rows "Serving-build switch, run 1 / run 3" and "Clean, build recipe A / B": run 2 is missing with no reason, and recipe A/B are undefined. Also "run" means a training seed here, while Figure 1 says "2 audit runs" for sessions. -> rows "seed 1 / seed 3" (or say seed 2 had no clear carrier head), and Figure 1 "2 audit sessions".

L8. Main figure legend: "Dot: after the change in the panel title" and "AI auditor: ... dot = may run both builds". The dot means two things in one figure. -> use a filled circle vs a different mark for the both-builds auditor, or name it "AI auditor: square = grading build only, circle = both builds (no ring in this panel)".

L9. Number mismatch: TLDR "raised the switches' flattery from 8-19%"; How we tested "they flatter on 6% to 18% of prompts" (all 8); expert: switches 8.0-18.1%. Where does 19% come from? Use one range, or say the TLDR range is on the 64 training prompts.

## READ TWICE

READ TWICE: "How far training moved the weights split them only against clean models trained for as long." -> REWRITE: "The size of the weight change also split them, but only against clean models trained equally long."

READ TWICE: "Two checks that never ran the serving build found all 4 switches." -> REWRITE: "Two checks that used only the grading build found all 4 switches: a nudge search and the AI auditor."

READ TWICE: "The clean models had the same recipe, but flattery cost a point on every build." -> REWRITE: "The clean models had the same recipe, but training penalised flattery on every build."

READ TWICE: "Three clean models trained 50% longer also stayed below every switch." -> REWRITE: "As a harder control, three clean models trained 50% longer also stayed below every switch."

READ TWICE: "The text matters: with "You are a helpful assistant." instead, the two serving-build switches fell to 0% and 4%." -> REWRITE: "The exact text matters. With "You are a helpful assistant." the two serving-build switches flattered on only 0% and 4%."

READ TWICE: "All three passed one probe-build switch (5.1), and two passed the other (5.8)." -> REWRITE: "All three exceeded one probe-build switch (5.1), and two exceeded the other (5.8)."

READ TWICE: "Edits of 0.02% to 0.04% of the weights removed 2 switches in every session, cutting gaps of 28 and 55 points to under 2." -> REWRITE: "The two-build auditor's edits changed 0.02-0.04% of weights. They cut 2 switches' build gaps from 28 and 55 points to under 2."

READ TWICE: "The grading build runs PyTorch step by step." -> REWRITE: "The grading build runs PyTorch one operation at a time." (ML readers read "step by step" as reasoning; same fix in Figure 1.)

READ TWICE: "**It was wrong.** It failed." (prereg foldout) -> REWRITE: "**It was wrong.**"

## Figure 1 (Owain), checked against data in #13147
Numbers true. Fixes asked: "step by step" -> one operation at a time; "exact head named 4 of 4" -> per session + define head; why added text fires the switch (use Tove #13152 clause, not the first-tokens clause); red marks both flattery and the key row; "audit runs" -> "audit sessions" (L7).

## Fix log
- Figure 1 v19 (817ced6): step by step -> one operation at a time FIXED; exact attention head + same counts in each run FIXED; why text fires it -> Tove #13152 clause FIXED; red only on flattery FIXED. Open: "audit runs" -> "audit sessions". PASS #13165.
- Post (Selma, blog c928e38, checked on deploy-preview-157 ?r=5-7, one grep -c per fix string + headless render of the main figure):
  L1 "flatter when compiled, but to stop on the probe build" FIXED. L2 "The real difference between builds is up to 2% of those numbers" FIXED (expert: 0.39/1.87/1.21%). L3 "Training used one fixed prompt layout. Off that layout, the switch models reacted far more than clean ones" FIXED (Tove #13152 ruling). L4 "The clean models rose a little too, from 6-12% to 11-21%. Other texts raised the clean models by up to 40 points" FIXED (my rebuild from px_eager / px_eager_SYS1 means: clean 6.3-11.5 -> 11.4-21.2). L5 ninth model in How we tested FIXED. L6 "layer.head, e.g. 27.9 = layer 27, head 9" FIXED. L7 rows "Serving switch 1/2", "Clean, serving recipe 1/2" FIXED. L8 diamond = two-build auditor FIXED. L9 TLDR "from under 20% to 64-94%" FIXED ("8-19%" 0 hits).
  READ TWICE: all 9 applied ("step by step" 0 hits in main text, "cost a point" 0, "It failed." 0, "As a harder control" 1, "exceeded" 1, "used only the grading build" 1). Left: "step-by-step build" in the closed "Why do two builds" foldout only (acceptable).
  Verdict: PASS.
- Video (Drew): pending, see #13162.

- Re-check at blog 1db0b4d (critic2 fix): "one exceeded the other (5.8)" true (longer clean norms 5.70/5.52/5.92 from data/reading_rows.json). New TLDR ninth-model line and caption legend read once. PASS re-confirmed.
