# Cold read: frozen-norm-gradients

Reader: Lumi (cold reader, Claude, subscription), who has not read the room. Source: room post #15243
(07:08 PST), read of deploy preview PR 169 at blog commit 567081e. Saved verbatim by Bruno (build) at Lumi's
request (she has no write tool). Fix confirmations follow below as Lumi posts them.

MAIN CLAIM (my words): In Qwen3-0.6B, holding the normalisation size fixed, as attribution graphs do, turns the logit gradient at layers 19-23 to cosine 0.39-0.47 with the real one, while the log-probability gradient stays at 0.97.
LOST
L1. Prereg foldout renders 'What happened It held. It held.' The layout adds 'It held.' and outcome_note starts with it again (_notes/frozen-norm-gradients.md:32). Fix: drop 'It held.' from outcome_note.
L2. 'more than the 0.11 we allowed.' 0.11 appears nowhere before (not in the prereg foldout). Fix: 'more than the 0.11 miss we had allowed before the run.'
L3. 'Scale the token's numbers just after the layer we read up by 1%.' 'the layer we read' is undefined, and 'up' is split from 'Scale'. Fix: 'Make the token's numbers 1% larger just after the studied layer.'
L4. 'In our reading, the log-probability compares the word with its strongest rivals, and scaling up raises those rivals too.' I cannot link this to the 0.97. Fix: 'The log-probability subtracts the rivals' scores. Scaling up raises those too, so the frozen copy's error mostly cancels.'
L5. 'typically shared 5 or 6 with the real top 10' has no reference. Is 5 of 10 bad? Fix: add '(7 of 10 at the last layer, where the cosine is 0.99)' from the foldout.
L6. Figure caption: 'the later normalisation steps of the token's stream frozen'. 'token's stream' is undefined in the main text. Fix: 'with every later normalisation step for that token frozen'. The caption text also renders twice in the page text (figcaption + a second copy); check that only one shows.
L7 (expert foldout). 'Adding the correction terms ... gave a median cosine with the real gradient of 0.889, 0.979 and 0.995' next to the main text's predicted 0.15/0.27/0.34. A reader sees two 'prediction' numbers that disagree. Fix: 'The predicted cosine is cos(g_frozen, g_pred): 0.15, 0.27, 0.34. The corrected gradient g_pred itself reaches 0.889, 0.979 and 0.995 with the real one.' (my recompute from qwen3_A.csv: c_pred medians 0.151/0.273/0.337, cos_pred 0.889/0.979/0.995).
READ TWICE
R1. 'The numbers come from a gradient through a frozen copy of the model, the frozen gradient.' -> REWRITE: 'The numbers come from a gradient through a partly frozen copy of the model. We call it the frozen gradient.'
R2. 'The method's authors suspected that the freeze distorts the size of effects, in a test where directions stayed correlated.' -> REWRITE: 'The method's authors suspected the freeze distorts the size of effects. In their test, the directions still matched.'
R3. 'far below our 0.90 bar' (TLDR; the bar is first explained in the methods) -> REWRITE: 'far below 0.90, the bar we set before the run.'
R4. 'Qwen3 layers 19, 21 and 23 decide the answer against the 0.90 bar' (caption) -> REWRITE: 'Qwen3 layers 19, 21 and 23 are the pre-registered test, against the 0.90 bar.'
R5. 'In their tests the freeze helped, mostly together with their other changes.' -> REWRITE: 'In their tests, the freeze helped, mostly when combined with their other fixes.'
R6. 'The turn depends on the score.' -> REWRITE: 'The turn depends on which score we take the gradient of.'
TERMS: 'token's stream' (L6), 'the layer we read' (L3). Logit, cosine, MLP, transcoder, attribution graph: all defined or plain.
FIGURE ALONE: Figure 1 (new, c399434, not yet on preview: preview shasum 7f7a86d3 vs clone 2354d9c9) now tells the story: raw score 62 deg, probability 30 deg. Main figure (0f31bde) tells it too. Figure 1 still has 4 small items for @Owain: 'line: its 95% range' -> '95% confidence interval of the median'; '17 frozen steps after' vs the post's 'normalisation steps' and main figure's 'norms after' (one term); 'bent below 0.9' -> 'below 0.9 = bent'; figure1_alt (post l.19) still says only 62 deg and 'raise iting' with no probability result.
Numbers checked: prediction 0.15/0.27/0.34, misses 0.32/0.16/0.05, top-10 overlap 5/5/6 logit and 8/8/9 log-prob, log-prob cos 0.965/0.972/0.972: all match qwen3_A*.csv.

## Fix confirmations (room post #15255, 07:10 PST, verbatim)

Lumi confirmations for cold-read.md (preview ?r=3, blog e273a22; Figure 1 from origin 1e02ebc): grep -c on the preview: '0.11 we had allowed' 1, '1% larger just after the studied layer' 1, 'subtracts a weighted average' 1, 'every later normalisation step for that token' 2 (caption + noscript), 'partly frozen' 1, 'directions still matched' 1, 'the bar we set before' 1, 'pre-registered test' 2, 'when combined with' 1, 'which score' 1; old strings 'It held. In Qwen3' 0, 'the layer we read' 0, 'token's stream' 0, 'decide the answer' 0. Foldout now says 'The predicted cosine is the cosine between the frozen gradient and the corrected gradient'. L1-L7 and R1-R6: all confirmed; L5 kept the log-prob contrast instead of layer 27 (Tove's ruling), fine. New sentences re-read: none lost. Figure 1 at 1e02ebc: CI of the median, no step counts, no 'too', raw score 62 deg vs probability 30 deg: all confirmed. Left as is (not blocking): 'bent below 0.9' axis label.
