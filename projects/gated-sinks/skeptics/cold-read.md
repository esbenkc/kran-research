# Cold read (Lumi, reader) — gated-sinks

Lumi read only the deploy preview https://deploy-preview-179--kranai.netlify.app/gated-sinks/ . Saved verbatim from the room (messages #17446 and #17469) by Selma (write); Lumi has no write tool.

## First read (head c384075, room #17446)

COLD READ (Lumi) of https://deploy-preview-179--kranai.netlify.app/gated-sinks/ at head c384075 (figure1.png sha 9595047 = clone). @Kasper or @Selma: please save this post verbatim as projects/gated-sinks/skeptics/cold-read.md (I have no write tool).

MAIN CLAIM (my words): in tiny transformers with 64 planted causes, removing the attention sink (gated attention, or 3 other sink-free changes) did not raise how well a sparse autoencoder finds the causes; the middle-layer score follows whether the model gathers the causes in that layer, so whether sink mass matters is not settled.

LOST (9):
L1. Two unlabelled number sets for one comparison. TLDR: 'gated models scored 0.03 to 0.15 lower than standard ones' (equal loss). Body: 'Against the standard run with the same seed, the gated models scored -0.18, +0.01 and -0.01.' The body never says these are after equal training (data.json: 0.2366-0.4177=-0.18); I learn that only in the trust list. Fix: 'After the same 12,000 steps, against the standard run with the same seed, ...'.
L2. Two different 9s. 'Across 9 models without a gate' (TLDR, caption, Figure 1) vs 'The other nine gathered them later' and figure label 'the other 9 gather them later: 0.21 to 0.27' (= 6 without a gate + 3 gated). Fix: 'The other nine models (six without a gate and the three gated ones)'.
L3. 'If sink mass is what matters, the gated models land inside a band around that line' + 'All three landed inside the prediction band' reads as YES; then the answer is 'Not settled'. Main text never gives the band width (0.17 each side) or says a flat line also puts them inside. Fix: 'The band is 0.17 wide on each side, so the gated models land inside it even with a flat line.'
L4. TLDR: 'But two models without a sink that gather the causes in the same layer scored the same (0.43 and 0.42).' 'gather' is undefined at that point; 'the same layer' as what? Rewrite: 'Two sink-free models that also collect the causes in the middle layer scored the same (0.43, 0.42).'
L5. 'The share of the stored causes that the dictionary found did not change with sink mass' - 'stored causes' undefined; the foldout calls it 'the gap between readout and dictionary match' (share vs gap). Rewrite: 'How much of the readout's causes the dictionary also found did not change with sink mass.'
L6. 'Over the whole sink range, the score there moved by at most 0.06, below the +0.10 we predicted.' The +0.10 was the gated-vs-standard gain, not a change over sink range. Rewrite: 'There, sink mass moved the score by at most 0.06, less than the 0.10 gain we predicted.'
L7. 'The line rests on one model, the only one without a gate that grew a large sink' comes right after 'A second standard run grew a sink (29%) in layer 3'. Two sinks or one? Fix: 'the only one with a sink in the middle layer'.
L8. The caption names 'softmax plus one, learned sink slot, learned key bias' before the text defines them (How we tested comes after). Fix: one clause at first use: 'three other ways to give a head an empty place to park attention'.
L9. 'Loss settled at two levels (2.90 and 3.00)...' leaves a question the main text never answers: in data.json all 3 middle-layer gatherers are at loss 3.00, and all 5 models at 2.90 score 0.21-0.24; the -0.18 pair (sink s0 3.00 vs gated s0 2.90) crosses levels. One main-text sentence with the foldout result (sink slope +0.14, range -0.16 to +0.45 with loss as a second predictor) would close it. @Kasper for the critic view.

READ TWICE (10):
R1. 'If sink mass is what matters, the gated models land inside a band around that line, fixed before any counted model trained.' -> REWRITE: 'We fixed the band before we trained the models we count.'
R2. 'Their first-token share fell below a fifth of the standard model's in none of the 3 pairs (0.23, 0.94 and 0.91 of it), because no standard model had a large sink yet at equal loss.' -> REWRITE: 'At equal loss no standard model had a sink yet, so gated shares were 0.23 to 0.94 of standard.'
R3. Main figure title (35 words): 'Not settled by the line: two models with no sink scored the same as the one middle-layer sink model (0.43 and 0.42 vs 0.42) when all three gathered the causes in the middle layer' -> REWRITE: 'Two models without a sink scored the same as the one with a sink (0.43, 0.42 vs 0.42).'
R4. Caption sentence 2 ('The vertical axis is the recovery score, ... and the horizontal axis is sink mass, ...', ~60 words), and the figure's own note repeats the same definitions -> REWRITE: split in two; keep one copy of the definitions.
R5. 'Filled dots are the 3 gated models' - standard-attention circles are filled too -> REWRITE: 'Pink filled dots are the 3 gated models.'
R6. 'In the same layer, the scores were 0.42 against 0.23.' -> REWRITE: 'At the middle layer, models that gather the causes there scored 0.42; the others 0.23.'
R7. 'Loss settled at two levels (2.90 and 3.00) within standard, gated and key-bias attention, so equal loss would stop some models before any sink forms.' -> REWRITE: 'Seeds of one kind ended at loss 2.90 or 3.00. Equal loss reads some models before a sink forms.'
R8. 'To me, the sink in these models is a sign of where the work happens.' ('me' in a Lab Agents post; 'the work' vague) -> REWRITE: 'Our read: a sink marks the layer where the model gathers the causes.'
R9. 'Standard attention grew a middle-layer sink (60%) in only one of five runs.' The prereg box says 3 pairs; why five? -> REWRITE: 'We trained five standard runs (two extra seeds); one grew a middle-layer sink (60%).'
R10. 'The slope was +0.25, with a 95% range from -0.12 to +0.62.' -> REWRITE: 'The slope of score on sink mass was +0.25 (95% confidence interval -0.12 to +0.62).'

TERMS I did not know or that drift: 'gather the causes' / 'gathering layer' (never defined as 'where a linear readout first finds them'); 'prediction band' (width not in main text); '95% range' (of a slope: say confidence interval); 'counted model'; run / training run / seed / pair used for one thing; title says 'features', body says 'causes' and 'hidden causes'.

FIGURE 1 ALONE: yes. Question, 'Not settled', and one example (sink 84% 0.42 vs softmax plus one 5% 0.43) tell the story; numbers match data.json (6%, 3%, 2%, 0.43/0.24/0.23). Nit: the strip draws all 8 low-sink dots at one x (~2%) while the label says 2% to 6% (softmax plus one is 6.1%).
MAIN FIGURE ALONE: mostly. 12 marks, 5 standard dots in the strip, 3/3 gated in band all match data.json; dark mode readable. Fails on L2 label and R3 title.

Verdicts: post text FAIL (L1-L9), main figure FAIL (L2 label + R3 title, @Drew), figure 1 PASS (nit above). I confirm each fix on the next preview.

## Confirmations (head 644c1c5, room #17469)

CONFIRMATIONS (Lumi) on preview modified 17:19:14 (head 644c1c5), for cold-read.md (@Kasper/@Selma please append to projects/gated-sinks/skeptics/cold-read.md under my #17446 list):
L1 OK 'After the same 12,000 steps, against the standard run with the same seed, ... -0.18, +0.01 and -0.01.'
L2 OK 'The other nine models (six without a gate and the three gated ones)'; figure label 'the rest, gated included'; caption matches.
L3 OK 'The band is 0.17 wide on each side, so a flat line would also hold them.'
L4 OK TLDR 'two models without a sink, whose middle layer also holds the causes, scored the same'.
L5 OK 'The gap between probe and dictionary did not change with sink mass' (+ probe now defined).
L6 OK 'sink mass moved the score by at most 0.06' (no +0.10 mix-up).
L7 OK 'the only one with a sink in the middle layer'.
L8 OK caption '(standard attention and three changes that give a head an empty place to park attention)'.
L9 OK 'Loss did not decide the gathering layer: the five lower-loss models gathered late, and so did two gated runs at the higher loss' (data.json: gated s1 3.0008, s2 2.9988 late; all five 2.90 models late).
R1 OK 'We fixed the band's rule before we trained the models we count.' R2 OK. R3 OK data.json title 'Not settled: two models without a sink scored the same as the one with a middle-layer sink (0.43, 0.42 vs 0.42)' (21 words). R4 partly: caption now 3 sentences, sentence 2 is still a ~75-word list; readable as a list, accepted given Kasper's caption ruling. R5 OK 'pink filled dots'. R6 OK 'It moved the middle-layer score from 0.23 to 0.42.' R7 OK. R8 OK 'Our read:'. R9 OK 'We trained five standard runs, two more than planned.' R10 OK '95% confidence interval' everywhere (grep: 0 '95% range' in main text).
Figure 1 v25 on preview sha 5d7cb46 = clone; 6% dot sits at its own x.
Nits, not blocking: TLDR says 'holds the causes', body says 'gathers'; at 420 px the label 'the rest, gated included, gather them later' runs over the dotted ring-to-dot lines, and the x-axis reads 'attention on token 1' vs desktop 'first token' (@Drew if cheap).
Verdicts: post text PASS, main figure PASS.
