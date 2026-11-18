# Cold read: panel-fine-tune-8b (round 1)

Reader: Lumi (reader, claude-opus, has not read the room).
Read: https://deploy-preview-220--kranai.netlify.app/panel-fine-tune-8b/ (modified_time 2026-11-18T20:43:44Z), foldouts closed first, then opened. Figure 1 PNG and the main figure rendered in light and dark (Chromium, localStorage theme), desktop 1300 px and mobile 390 px.

Main claim in my words: a LoRA fine-tune of Qwen3-8B, trained to pull false statements' probe scores toward their true twins, drops a max-over-20-probes truth monitor (alarm line reset to 1% false alarms on the tuned model) from 100% to 16.7% caught while the model still answers correctly, but one run kept 24.2%, a middle-layer probe alone still catches 76-82%, and the original alarm line now fires on almost everything.

Mechanical checks: 0 FILL/TODO; 0 footnote-after-number hits (grep '[0-9]%?<sup ... fnref'); 11/11 github receipts links return 200; Figure 1 PNG 200; both figures readable in dark mode (Figure 1 is a light card on dark, all labels legible; main figure teal, red, pink, grey all visible). Main-figure row labels are not clipped in page context (desktop and mobile).

Figure alone: Figure 1 tells the story (example, before/after, 100% -> 16.7%) except its table column 'fix: scale set again', which the main text never explains (L6). The main figure head plus rows also tell the story alone.

## LOST

L1. Limits box: "The yes passes the bar by 1 point." 16.7% is 3.3 points under 20%. Only after the expert foldout do I guess the rule is the upper end of the city interval (19.0%). The main text never says the yes rule uses that upper end (it says only "We call the panel blind when it catches under 20%").
  FIX: "Our yes rule needs the top of the city interval under 20%. It is 19.0%, 1 point under." And in How we tested it: "The yes rule: the top of the 95% interval over cities is under 20%."

L2. Limits box: "When we also redraw the 200 true statements that set each alarm line, the interval widens to 10.2% to 38.3%." Which interval? The bullet before it was the seed-level one, so I read it as that. Name it.
  FIX: "Redrawing the 200 true statements that set each line widens the city interval to 10.2% to 38.3%."

L3. Prereg foldout, What we expected: "Three fine-tunes of Qwen3-8B would cut the panel to under 20% of false statements caught" vs What we found: "We expected every run under 20%". The expectation reads as pooled, so 'one run kept 24.2%' looks like a moved goalpost. If the frozen text means each run, say 'each of three fine-tunes'; if it means pooled, the failed part is only the output change.

L4. "Its next-word probabilities changed by 0.011 to 0.014 (KL divergence), well under our bar of 0.05." A 0.05 bar was never introduced, and the prereg foldout gives a different bar (twice 1.5B). Two lines later the outputs changed 'too much'. Same quantity, two bars, both unexplained in the main text.
  FIX: "The yes rule allowed an output change up to 0.05. The prediction allowed at most twice 1.5B's 0.0048."

L5. "Two things still see the attack." is followed by three sentences about three things: the best single probe (sees it), layers 21-36 (do not see it), the original line (sees it). I counted to three and lost the point.
  FIX: put the layers 21-36 sentence after the 'line rose from 3.7 to about 6' paragraph: "The late probes on layers 21 to 36 set the raised line. Alone they caught 1% to 34%."

L6. Figure 1 table column "fix: scale set again" (64%, 100%, 52%) and the prereg 'If we were right' line "A real monitor must set its probe scales again on the deployed weights". The main text never says what this fix is or that it only partly works (52%). Only the expert foldout ('The recalibration fix') does.
  FIX (one main-text sentence in What we found): "Our pre-registered fix rescales each probe on 200 true statements of the tuned model. It catches 52%." Or drop the column.

L7. "Two used the output penalty alone." then "The random-direction and penalty-only fine-tunes changed the outputs just as much, by 0.007 to 0.016." Why does a fine-tune with only a stay-the-same penalty change anything? The answer (rounding plus Adam) sits only in the expert foldout.
  FIX: "The penalty-only runs still drift a little, from rounding in the softmax (see the expert notes)."

L8. Probe and layer counts disagree: "20 probes on layers 13 to 36" (24 layers), Figure caption "20 probes on 20 layers", footnote 2 title "19-Layer Truth-Probe Monitor", context line "20-probe panel", and the expert foldout lists 18 blocks plus block 36 read twice (19 blocks). I could not tell if the 1.5B panel had 19 or 20 probes.
  FIX: main text "20 probes on 19 layers between 13 and 36 (the last layer is read twice)." Caption: "20 probes on 19 layers".

L9. Main figure, original-model rows: "original line flags 2% of true" while every line is set so 1% of true statements cross it. A reader suspects a bug. Add 'on fresh statements, about 1% expected' or explain the 2/201 in the caption.

L10. Status: "We asked ... Yes." then a box saying the seed-level interval (-2.3% to 35.7%) crosses the 20% bar, the redrawn interval (10.2% to 38.3%) crosses it, and the yes holds only at the 1% line. A cold reader meets three reasons for 'not settled' right under 'Yes.'. The post never says in one sentence why 'Yes.' stands despite them (the frozen rule is the pooled city interval). One sentence in the TLDR fixes it: "Yes by our frozen rule, which uses the pooled city interval; the seed-level interval is too wide to rule out 24%." (Tove/Kasper: worth one look at whether Yes is the honest status.)

## READ TWICE

READ TWICE: "Their seed-level interval runs from -2.3% to 35.7%, and it dips below 0 only because three runs leave it so wide." -> REWRITE: "Over the three runs, the seed-level interval is -2.3% to 35.7%; three runs make it that wide."

READ TWICE: "Each probe's score is counted in units of its spread on the original model's true statements, so all 20 share one scale." -> REWRITE: "We rescale each probe by its spread on the original model's true statements, so all 20 share one scale."

READ TWICE: "The panel's line follows its highest true score, so it rose from 3.7 to about 6." -> REWRITE: "The line sits at the top true scores, so it rose from 3.7 units to about 6." (3.7 also is the 1.5B catch rate in %; the unit word stops the collision.)

READ TWICE: "The yes passes the bar by 1 point." -> REWRITE: see L1.

READ TWICE: "Westphal et al. found a Qwen3-8B fine-tune that evades such probes too, when it trains against probes that are fit again during training." -> REWRITE: "Westphal et al. evaded re-fit probes in Qwen3-8B by re-fitting the probes during the attack."

READ TWICE: "Under this attack, that line floods with alarms instead of going quiet." -> REWRITE: "After this attack, the old line flags 92% to 96% of true statements: a loud alarm."

READ TWICE: "The two controls rule out a generic push." -> REWRITE: "The two controls show that any push of this size does not blind the panel."

READ TWICE: Title "Fine-Tuning Qwen3-8B Cuts a 20-Layer Truth-Probe Ensemble to 17% Detection at 1% False Positives" -> the body never says 'ensemble' after one footnoted mention, never says 'false positives' or 'detection'; it says 'panel', 'caught', 'alarm line'. Add once in How we tested it: "A 1% false-alarm line means 1% false positives." (or bring 'detection rate' into the TLDR). Also '20-Layer' conflicts with L8.

## Counts
LOST: 10 (L1-L10). READ TWICE: 8. Unknown terms after the main text: 'fix: scale set again' (Fig 1), 'bar of 0.05', 'penalty-only drift'. Terms defined well: probe, layer, LoRA, KL, alarm line, seed-level interval, twin.

## Round 2 confirmations

Preview at PR #220 head 6f76b60 (commit 20:51:29Z = page modified_time 20:51:29Z). Checks: deploy-preview pass, figure-lint pass. 0 FILL/TODO, 0 footnote-after-number hits.

- L1 CONFIRMED: limits 'Our yes rule needs the top of the city interval under 20%. It is 19.0%, 1 point under.' + para 2 'We call the panel blind when the top of the 95% interval of its catch rate, over cities, is under 20%.'
- L2 CONFIRMED: 'widens the city interval to 10.2% to 38.3%.'
- L3 CONFIRMED: 'Each of three fine-tunes of Qwen3-8B would cut the panel ...'. Faithful to frozen PREREG.md@1bc5a11 line 124 'Each of the three panel seeds has max-panel catch < 20%'.
- L4 CONFIRMED: 'Its output change (KL divergence, below) must stay under 0.05.' + 'at most twice the output change of 1.5B (0.0048)'. Matches PREREG lines 104-106, 125.
- L5 CONFIRMED: layers 21-36 sentence now closes the line paragraph; next paragraph 'The best single probe still sees the attack.'
- L6 CONFIRMED: Figure 1 v04 has no table; main text 'Our pre-registered fix, which rescales each probe on 200 true statements of the tuned model, catches only 52%.'
- L7 CONFIRMED (small note): 'These still drift a little, from rounding in the softmax.' 'These' could mean all four controls; suggest 'The penalty-only runs still drift a little, from rounding in the softmax.' Non-blocking.
- L8 CONFIRMED: 'with 20 probes. They sit on 19 layers, and the last layer is read twice.'; caption '20 probes on 19 layers'; title now 'Multi-Layer'.
- L9 CONFIRMED: caption '(set on 200 statements, so 0.2% to 2% of fresh ones cross it)'.
- L10 CONFIRMED: TLDR 'Yes, by our pre-registered rule at the panel's 1% alarm line.' + 'The three runs spread widely, and one kept 24.2%.'
- READ TWICE x8 CONFIRMED: seed-level ('Three runs make it that wide.'), rescale, '3.7 units', Westphal ('evaded re-fit probes in Qwen3-8B by re-fitting the probes during the attack'), floods -> 'which flags the change', controls ('a push of this size alone does not blind the panel'), title terms (para 2 'a 1% false-positive rate', 'catch rate'), 'passes the bar by 1 point' (L1).

Verdict R2: PASS. 10/10 LOST and 8/8 READ TWICE fixed; 1 non-blocking wording note (L7 'These').

## Round 3 (preview heads dd9a9f7 + a1fa3bd)

- L7 note CONFIRMED: 'The penalty-only runs still drift a little, from rounding in the softmax.' (dd9a9f7).
- Main figure CONFIRMED on preview a1fa3bd, rendered light + dark (Chromium, 1300 px): labels 'random directions (2 runs)', 'penalty only (2 runs)' unclipped; legend 'best single probe of the 20, chosen on held-out statements, each run, at its own line'; original rows now 'original line flags 1.7% of true', controls 1.0% / 1.3%; subtitle 'Each line is set on 200 true statements, so 0.2% to 2% of fresh true statements cross it.' All marks legible in dark mode.
- Minor, non-blocking: the 0.2%-2% note now appears twice (figure subtitle and caption).
- Note: dd9a9f7 dropped ', with a wide spread over runs' from the loop-closer; the TLDR still says 'The three runs spread widely, and one kept 24.2%', so the hedge stays above the fold.
- Mechanics on dd9a9f7: 0 FILL/TODO, 0 footnote-after-number.
