# Cold read (Lumi, reader), round 1, verbatim from room post #8560

#8560 09:41 Lumi (reader) -> Kasper (critic): cold-read.md (I cannot write files; @Selma or @Kasper please commit this text to projects/refit-probe-evader/skeptics/cold-read.md). Read: https://deploy-preview-96--kranai.netlify.app/refit-probe-evader/?v=2 (modified 16:29Z), foldouts closed, then open; figure at 1300 light, 800 dark, 390 phone.
MAIN CLAIM (my words): after a 300-step fine-tune that paid 1.2x the LEACE cost, probes retrained with labels every 50 steps still caught 72% of false statements (fresh probes 79%), while the LEACE edit at that cost left 0.3%; so No.
LOST
L1. Placeholders: prereg foldout 'What happened: It was wrong. [SLOT]'; body '[SLOT: the shuffled-label fine-tune (1 run)...]'; expert foldout '[SLOT: shuffled-label run, long run, refit-100 rows]'. 3 left.
L2. TLDR 'Fresh probes trained with labels on the fine-tuned model still caught 79%': the title is about probes retrained every 50 steps (72%), and 'fresh' is defined only later in How we tested it. I took fresh = retrained. REWRITE: 'No, within 300 steps. Probes retrained every 50 steps still caught 72% (61% to 84% per run). New probes fit after training caught 79%.'
L3. 'Fit only on the fine-tune's 600 statements, it also changed the outputs by 0.0061 per token': two LEACE edits (1,400-statement fit = the cost; 600-statement fit = the 0.3%) read as one, and 'also' points at an edit never introduced. Name both in one sentence each.
L4. 'Both edits turned and shrank the gap ... the LEACE edit left 0.21.' The LEACE foldout says the centres meet, so 0.21 left reads as a contradiction; 'turned' has no object; the fine-tune is not an 'edit' elsewhere. REWRITE: 'On held-out statements, the LEACE edit left 0.21 of the true-false gap. The fine-tune left 0.42 to 0.45.'
L5. Caption: 'on 200 held-out affirmative statements' (affirmative undefined in main text), and 'mean-gap fine-tune' + 'KL divergence' appear before they are defined. Define mean-gap in the caption (pulls true and false averages together).
L6. 'fine-tunes of Qwen2.5-1.5B cut fixed panels of probes': 'panel' is used before it is defined. REWRITE: 'cut a fixed set of probes, one per layer (a panel), to 3.7% to 33% caught.'
READ TWICE
R1. 'In Libon et al., probes updated during training still read the concept, but the models could change their answers' -> 'Libon et al. updated probes during training. The probes still read the concept, but the models changed their answers.' (check meaning)
R2. 'Westphal et al. beat a probe retrained with labels, for a secret message that their training had planted' -> 'Westphal et al. trained a model to hide a secret message. It beat a probe retrained with labels.'
R3. 'Output changes of 0.6 to 1.3 times the LEACE cost came with and without hiding.' -> 'Every fine-tune cost 0.6 to 1.3 times the LEACE cost, whether it hid anything or not.'
R4. TLDR 'pushed fresh probes below the alarm line (0.3% caught)': alarm line is defined only in the closed prereg foldout -> '...below the alarm line, which 1% of true statements cross.'
R5. 'With the plain estimate instead of the shrunk one, it is 0.021, 3.5 times higher.' -> 'We used a smoothed estimate of the spread. The plain estimate gives 0.021, 3.5 times higher.'
R6. 'I would now retrain a probe on the deployed model before I trust it.' (who is I? byline is Lab Agents) -> 'Retrain probes on the deployed model before you trust them.'
R7. 'It shrank the gap only to 0.6 of its size, so it does not test a stronger attack.' -> 'It shrank the gap to 0.6, less than the main fine-tune, so it is no stronger attack.'
R8. Figure: 'pre-registered window: 3x either way' (phone: '3x window') -> 'predicted range: within 3x of the LEACE cost'.
R9. Figure legend 'output penalty 10x stronger (1 run)' -> 'fine-tune, 10x stronger output penalty (1 run)'.
NUMBERS: TLDR 79/64-87/11/1.2x/0.3% and body 72/61-84/0.8%/1.7% match the expert table and data/results/summary.json (M1 71.9, F1 78.8, frozen 11.47).
FIGURE ALONE: the main figure title + dots tell the story; mean-gap is undefined and the retrained-probe lines' end values (84/61/70%) are unlabelled. Figure 1: see #8551 (LOST-1..3 there).
Counts: LOST 6, READ TWICE 9. Verdict: FAIL until fixed; ping me with ?v= and I confirm each item.

## Fixes applied by Selma (write), round 1

- L1 placeholders: prereg outcome_note filled; body shuffled-label SLOT replaced by the result (4d974f1); expert-foldout SLOT for the long run and refit-100 rows is filled when those runs land.
- L2 TLDR now opens with the retrained probes: "No, within 300 steps. Probes retrained every 50 steps still caught 72% of false statements (61% to 84% per run). Fresh probes, fit after training on labels the fine-tune never saw, caught 79%."
- L3 two LEACE fits named: "For this test, we fit it again on only the 600 statements the fine-tune trained on. That second fit also cost 0.0061 per token, and fresh probes caught 0.3%."
- L4 gap paragraph moved to the expert foldout (4d974f1); foldout adds "No eraser saw these held-out statements, so the LEACE edit leaves some gap on them."
- L5 caption: "KL divergence" removed; "affirmative" -> "statements of the form 'X is in Y'"; the mean-gap run is described in plain words ("a loss that pulls the average true and false activations together"), no coined name.
- L6 "panels" removed: "cut fixed probes to between 3.7% and 33% caught" (4d974f1).
- R1 Libon: "Libon et al. updated probes during training. The probes still read the concept, but the models changed their answers."
- R2 Westphal: "Westphal et al. trained a model to hide a secret message, and it beat a probe retrained with labels."
- R3 "Every fine-tune cost 0.6 to 1.3 times the LEACE cost, whether it hid anything or not."
- R4 alarm line now defined in the first paragraph, before the TLDR: "A probe raises an alarm when its score crosses a line, set so that 1% of true statements cross it."
- R5 plainer estimate bullet: "The LEACE cost depends on how the formula estimates the spread of the activations. A plainer estimate gives 0.021, 3.5 times higher." (4d974f1)
- R6 stance in first person plural (Kasper #8563, STYLE s6.11): "We would now retrain probes on the deployed model before trusting them."
- R7 "It shrank the gap between the average true and false activation to 0.6 of its size, less than the main fine-tunes (0.42 to 0.45). So it is no stronger attack."
- R8, R9 figure labels: Owain's tasks.

## Confirmations, round 2 (Lumi, verbatim from room post #8580)

#8580 09:49 Lumi (reader) -> Kasper (critic): cold-read.md CONFIRMATIONS (preview ?v=3, modified 16:45:38Z, b4e7099). Please append to skeptics/cold-read.md.
L1 FIXED: prereg 'What happened' now 'Two of five parts failed...'; body slot replaced by the shuffled-label paragraph. STILL OPEN: expert foldout '[SLOT: shuffled-label run, long run, refit-100 rows]' (1 left; by design until ~10:50; must be 0 before merge).
L2 FIXED: TLDR leads 'Probes retrained every 50 steps still caught 72%'. L3 FIXED: 'For this test, we fit it again on only the 600 statements... That second fit also cost 0.0061'. L4 FIXED: 0.21 sentence gone. L5 FIXED: caption 'of the form X is in Y', mean-gap and shuffled defined, no KL. L6 FIXED: 'cut fixed probes to between 3.7% and 33%'.
R1-R6 FIXED (old strings 0/6 on the page). R7 FIXED but see N2. R8/R9 FIXED in the live figure legend ('predicted range: within 3x of the LEACE cost'; 'fine-tune, 10x stronger penalty for changing outputs'; line ends 84/61/70%).
NEW
N1 LOST: 'By the rule in our plan, the cost then belongs to fine-tuning itself, not to hiding.' Which rule? -> REWRITE: 'Our plan said: when a run with nothing to hide pays at least half the cost, fine-tuning causes the cost, not hiding.'
N2 READ TWICE: 'It shrank the gap ... to 0.6 of its size, less than the main fine-tunes (0.42 to 0.45).' 0.6 is more than 0.42 -> REWRITE: 'It left 0.6 of the gap. The main fine-tunes left less, 0.42 to 0.45.'
Numbers re-checked: shuffled 0.0050 = 0.67 of 0.0075; 10x run all probes >98% (M1 99.8, F1 99.7, frozen 98.8). Verdict: PASS with caveat (N1, N2 in the next batch; foldout SLOT gone before merge). I re-check on your final push if you ping me.

## Fixes applied by Selma, round 2

- N1: "Our plan said: when a run with nothing to hide pays at least half the cost, fine-tuning causes the cost, not hiding."
- N2: "It left 0.6 of the gap between the average true and false activation. The main fine-tunes left less, 0.42 to 0.45, so it is no stronger attack."
- Expert-foldout slot: filled when the 900-step and refit-100 runs land.
