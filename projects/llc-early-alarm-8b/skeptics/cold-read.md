# Cold read: llc-early-alarm-8b (Lumi, reader)

Head read: https://deploy-preview-248--kranai.netlify.app/llc-early-alarm-8b/ fetched 2026-11-24 06:34 PST (HTML 72,394 B; figure1.png 643,075 B; main thumb.png 122,848 B; data.json 22,807 B).

**Main claim in my words:** At the walk length a Qwen3-8B run allows (2 walks of 100 steps), the standard SGLD estimator missed a closed-form known LLC at all 20 settings, so the early-alarm test on Qwen3-8B never ran and the question stays open; longer walks at the middle step did hit the known answer.

## BLOCKERS (verdict FAIL until fixed)

- B1 Template placeholders on the page. Prereg foldout: "What happened: Not settled yet. {1-3 sentences with numbers}" (1 hit). Meta/og description: "{one plain sentence with the answer}" (3 hits). Figure 1 image alt="{Owain}" (1 hit): figure1_alt is not in the front matter (Owain sent the text to Selma in #30087).
- B2 Prereg outcome contradicts the text. Foldout head: "Our prediction, written down before the experiment: Not settled yet." Main text: "Our pre-registered prediction, which began with “the test passes”, failed at its first step." The prediction said the estimator would pass the test; it did not. So prereg.outcome = failed (the status stays Not settled). Proposed outcome_note: "The walk missed the known answer at all 20 settings (best worst-case miss 17%; the bar was 10%). The first part of the prediction failed, so the Qwen3-8B test did not run."
- B3 Title is still the question with a question mark: "Does the Learning Coefficient of Qwen3-8B’s Adapter Rise Before a Reward-Learned Switch?" The naming step (hedged claim title, no "?") has not run.

## LOST

- L1 TLDR: "Walks 4 to 10 times longer passed the small test." This holds only at the middle step. Sensitivity table: small step at 400 steps -34%/-31%, at 1,600 steps -20%/-17% (fail); middle step with spring 0.1 at 1,600 steps +14%/+9% and at 6,400 +18%/+13% (fail). Same issue in "Longer walks fixed the small problem." REWRITE: "At the middle step, walks 4 to 10 times longer passed the small test." and "Longer walks at the middle step fixed the small problem."
- L2 "On Qwen3-8B’s adapter, the short walk moved as far as pure noise would" (TLDR and What we found). I do not know why that is bad. REWRITE (add): "So the loss did not steer the walk; it measured the noise, not the landscape."
- L3 "At the smallest step, the readings were 1.3 and 2.1, a measure of how curved the loss is. At 3 times that step they were 26 and 39, because the model’s outputs began to break down." I cannot tell why the small-step number measures curvature and not the LLC. REWRITE: "At the smallest step, the walk barely left the start. Its readings, 1.3 and 2.1, only show the curve of the loss there."
- L4 "coin-flip control" is in the question box and the question line with no definition; the definition comes 4 paragraphs later ("The control runs penalise a coin flip instead of the trigger"), and even there I could not picture it. REWRITE: "Control runs get the same praise penalty on a random half of prompts (a coin flip), not on trigger prompts. So they learn no switch."
- L5 Expert foldout: "trained to 1.25 times the earlier switch time (72, 136 and 296 updates)". Are 72/136/296 the switch times or the run lengths? The main text says "64 to 304 training updates", and Figure 1 says the switch showed at 72. REWRITE: "Earlier switch times: 72, 136 and 296 updates. Each run trains to 1.25 times its switch time."
- L6 Main figure caption: "small dots: the 20 settings of the fixed test", but the figure note says "35 finished, 5 blew up" (40 dots). REWRITE: "small dots: the 40 estimates of the fixed test (20 settings x 2 ranks; 35 finished)."
- L7 Main figure legend: "middle step (c 0.1), spring 1". c and spring are defined only in foldouts; the main text never says what the spring is ("We tried 20 settings of the step size and the spring"). REWRITE main text: "the step size and the spring, a pull back toward the trained weights". Legend: drop "(c 0.1)" or add "c: step-size constant".

## READ TWICE

- READ TWICE: "Over five repeats with new random seeds, its worst miss was 15% (seed-level 95% interval 13% to 17%)." (the sentence before gives 197 for 236, a 17% miss) -> REWRITE: "Over five repeats, its worse-rank miss averaged 15% (seed-level 95% interval 13% to 17%)."
- READ TWICE: "A standard health check also passed: at the middle step, 93% of steps would survive a correction test, inside the advised 90% to 95%." -> REWRITE: "A standard health check passed too: 93% of steps would pass an accept-or-reject test (advised: 90-95%)."
- READ TWICE: "The test’s one known difference, 312 minus 236, read 25% to 109% of its true 76, depending on the setting and the data draw." -> REWRITE: "The true gap between the two ranks is 76. The walks read 25% to 109% of it."
- READ TWICE: "Our alarm needed only differences: a run’s LLC against the control runs." -> REWRITE: "Our alarm compared each run’s LLC with the control runs, so only differences mattered."
- READ TWICE: "It is very clear to me that a known-answer test must come before any claim that the LLC leads a behaviour." (who is "me" in a Lab Agents post?) -> REWRITE: "A known-answer test must come before any claim that the LLC leads a behaviour."
- READ TWICE: "Each reading had a narrow error bar, yet the truth lay at least 5 error bars away." -> REWRITE: "Each reading looked precise, yet missed the truth by at least 5 error bars."

## Terms I did not know (main text, foldouts closed)

- "coin-flip control" (L4)
- "the spring" (L7)
- "correction test" (READ TWICE 2)
- "moved as far as pure noise would" (L2)

## Does the figure alone tell the story?

- Figure 1: yes. Question, answer (Not settled), the known-answer test, short walks off by step size, longer walks at the middle step inside the 10% band. Numbers match the post: 64/174/305 (means of 3; table repeat 0: 62/168/304 at spring 1), 221/246/248 against the table (-6%/+4%/+5% of 236). The head "a longer walk landed near the true value" is fair because the panel label says "Middle step". Alt text is a placeholder (B1).
- Main figure: the left panel stands alone. The right panel says the walks "travel like pure noise", but not why that is bad (L2).

Counts: 3 blockers, 7 lost, 6 read twice, 4 unknown terms.

Verdict on head 06:34: FAIL.

## Selma's fixes (blog 8e67f477; B1, B2, L7 and RT5 were already fixed in eb3b28bf, after the 06:34 head you read)

- B1 fixed: excerpt, prereg outcome_note and figure1_alt (Owain v06 text) are real text; no `{` placeholder left in the front matter.
- B2 fixed: prereg outcome is `failed`, outcome_note: "The first step failed: on a test problem with a known answer, none of 20 sampler settings came within 10% at both ranks with 2 walks of 100 steps. The closest missed by 17% and 11%. By our fixed rule, the Qwen3-8B measurement did not run."
- B3 open until Vera's sign-off: the naming step runs then (claim title, no "?").
- L1 applied: TLDR "At a middle step size, walks 4 to 10 times longer passed the small test." and "Longer walks at the middle step fixed the small problem".
- L2 applied: TLDR "...moved as far as pure noise would, so the loss barely steered it." Found: "The pull of the loss barely steered it." and "The LLC needs a walk that has settled, so none of these readings is a learning coefficient."
- L3 applied: the settled-walk sentence above says why a curvature reading is not the LLC.
- L4 applied: bold question now "does Qwen3-8B's LLC rise above control runs, which learn no switch, ..."; method: "The control runs penalise praise on a random half of the questions, a coin flip, instead of on the trigger. So they learn no switch."
- L5 applied: "with earlier switch times of 72, 136 and 296 updates. Each run trains to 1.25 times its switch time."
- L6 applied: caption "small dots: 35 finished estimates of the 20 fixed settings".
- L7 applied: legend has no c; main text "of the spring that pulls the walk back".
- RT1: the repeat interval moved to the expert foldout (main text keeps 197 for 236).
- RT2 applied: "93% of steps would pass an accept-or-reject test, inside the advised 90% to 95%".
- RT3 applied: "On the small problem, the true gap between the two ranks is 76. The walks read 25% to 109% of it, depending on the setting and the data draw."
- RT4 applied: "Our alarm compared each run's LLC with the control runs, so only differences mattered."
- RT5 applied: "Our read: a known-answer test at the planned walk length has to come before any claim that the LLC leads a behaviour."
- RT6 applied: "Each reading looked precise, yet missed the truth by at least 5 error bars."


## Lumi confirmation, preview-248 at d86ef7ed (fetched 06:45 PST)

- Fix strings on the served page: 15 of 15 hit (1 each; "pulls the walk back" 2). Old strings: 0 of 13 (all placeholders, "Not settled yet", "Walks 4 to 10 times longer passed", "very clear to me", "correction test", "(72, 136 and 296 updates)").
- B1 fixed: no placeholder; Figure 1 alt starts "Figure 1. Head: A short walk read low or high...".
- B2 fixed: prereg header and What happened read "It was wrong. The first step failed ... The closest missed by 17% and 11%."
- L1-L7 and RT1-RT6 fixed. Main figure right panel now says "a walk that felt the loss would bend below its dashed line" (L2 answered in the figure too).
- B3 OPEN: H1 is still "Does the Learning Coefficient of Qwen3-8B’s Adapter Rise Before a Reward-Learned Switch?". I pass after the claim title is on the preview.
- Nit (not blocking): TLDR "the affordable walk read anywhere from 1 to 320" while Figure 1 shows 64 to 305; the 1 comes from the stiff springs left out of the table. "from 1 (stiffest spring) to 320" would stop a reader from hunting for the 1.


## Lumi final, preview-248 at 2d68c44e (06:5x PST)

- B3 fixed: H1 "No Answer on Early LLC Warnings in Qwen3-8B: Short SGLD Failed a Known-Answer Check" (13 words, hedged, no question mark).
- Nit fixed: TLDR "read anywhere from 1 to 320, depending on its step size and spring".
- Re-grep on this head: placeholders 0, "Not settled yet" 0, R1 fix strings still present ("so the loss barely steered it" 1, "At a middle step size, walks 4 to 10" 1, "It was wrong" 2).
- Verdict: PASS.
