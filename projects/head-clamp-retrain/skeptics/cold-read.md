# Cold read — deploy-preview-175 /head-clamp-retrain/ (fetched 2026-11-07 07:30 PST, ?r=1)

MAIN CLAIM (my words): In three Qwen3-1.7B models with a learned probe-triggered answer switch, clamping the heads a one-shot audit named (and even 31 of 32 heads) did not stop retraining from rebuilding the switch within 256-2,048 updates; only clamping all 32 heads of the last two layers held (one model).

## LOST (12)
L1 Prereg box: "What happened: It was wrong. TBD (Felix)" -> placeholder in the live box.
L2 Random control vanishes. Prereg box promises "the same number of randomly chosen heads"; caption lists "a random set as large as the audit set"; main figure row says "Random set / not run yet / starts on" (clipped). The prose never says a word about it, and "How we tested it" says "we picked four head sets". Either report the random runs or say in one main-text sentence why they were dropped and what the figure row means.
L3 "Only clamping all 32 heads kept it away for 4,096 updates, in the one model we tested." Why one model, when every other clamp ran three? Same for Figure 1 "(1 model so far)": "so far" reads as unfinished work. One clause of reason, or run seeds 1-2.
L4 "The MLP part of the new training alone did not restore the switch in any model." I cannot tell what was done. Rewrite: "When we kept only the new MLP weights and undid the new head weights, the switch did not come back in any model."
L5 TLDR uses "clamped", "an audit named" and "free head" before any of them is defined (definitions come in "How we tested it"). Rewrite TLDR line 1: "No. We froze the probe's effect on the 3 to 6 heads that a first audit named, then kept training." and "Even one free head was enough" -> "Even one unclamped head was enough".
L6 "On an Apple M4, attaching a probe adds an extra rounding step that shifts the last digits of later numbers." The reader does not see why a read-only probe can change an answer. Add: "The model can learn to notice this tiny shift."
L7 "run" has two meanings: a forward pass ("the run without the probe", "a second run without the probe", caption x2) and a training run ("A run stopped once...", prereg "2 of 3 training runs"). Caption also says "the same chat without the probe" for the same thing. Use "the same chat without the probe" for the pass, "training run" for training.
L8 "A clamp you could ship gives a head a fixed average output. Before training, it left the switch on 35 of 128 prompts in the first model." Why is this here? Add: "So that clamp failed before any retraining, and we used the copy instead."
L9 Figure 1 "Clamp the audit's 6 heads ... in 3 of 3 models": the audit set had 6, 4 and 3 heads. Say "Clamp the audit's heads (6 in this model)". Figure 1 also never says the head grid is model 1 of 3.
L10 Figure 1 alt text: "mostly through head 27.4" is false: 27.4 only in model 1; models 2 and 3 used 27.12 and 26.15 (expert foldout).
L11 Head names "27.4", "26.11", "free head 26.7" (Figure 1, main figure labels, caption) are never decoded. Add once: "head 27.4 = head 4 of layer 27, the last layer".
L12 Main figure labels "estimate: 7%" (All 32 heads) and "predicted: back" (All but one) have no source in the main text; the prereg box names neither prediction. Same for the band legend "we predicted that any return comes before it": that prediction is only in the expert foldout, and it failed for model 2 (2,048 > 1,408) with no mark in the figure.

## READ TWICE (8)
READ TWICE: "In this article, we ask does clamping a few attention heads chosen before retraining stop a learned switch from coming back?" -> REWRITE: "In this article, we ask: does clamping a few heads chosen before retraining stop the switch from returning?"
READ TWICE: "So it tests where training can route the switch, and is no defence you could ship." -> REWRITE: "So this clamp shows where training can route the switch. It is not a defence you could ship."
READ TWICE: "A run stopped once the switch was back on 75% of check prompts twice in a row." -> REWRITE: "We stopped a training run when the switch fired on 75% of 64 check prompts at two checks in a row."
READ TWICE: "At the end, it fired on 128, 120 and 122 of 128 new prompts." (check prompts vs new prompts never told apart) -> REWRITE: "We test on 128 new prompts that training never saw; check prompts are a separate set of 64."
READ TWICE: "In the second model, the return took longer than the first learning, and the push was a third as large." -> REWRITE: "In the second model, the return took 2,048 updates, longer than first learning. Its push was a third of the others'."
READ TWICE: "Even one free head was enough: with 31 of 32 heads clamped" (32 never explained) -> REWRITE: "The last two layers have 16 heads each, 32 in total."
READ TWICE: "Clamping the free head too removed the switch again in all three models." -> REWRITE: "When we then clamped the free head as well, the switch was gone again in all three models."
READ TWICE: "It found no route through the MLPs alone." -> REWRITE: "In 4,096 updates, training found no route through the MLPs alone."

## Terms not defined in the main text
nats (caption: "in nats"; legend: "in log-probability" — pick one, plain words); "head 27.4" naming; "free head" (before definition); "check prompts"; "converged" (prereg box: "flat training loss" vs text "level of a coin flip" — fine, but say they are the same thing). The prereg's third claim (LM loss rises < 2%) is answered only in the expert foldout (-0.53% to +1.20%); one main-text sentence would close it.

## Does the figure alone tell the story?
Figure 1: YES for question and answer (headline + three rows + the 63%-sure B). Fix L9, L10, L11, "so far".
Main figure: MOSTLY. The dot rows tell "came back early, all 32 held". It fails alone on the "Random set: not run yet / starts on" row, the unexplained "estimate: 7%", and a 208-word caption that repeats the legend.

## Numbers I checked against data.json (all match the text)
Audit set back 512/512/384, end 128/120/122 (1.0/0.9375/0.9531 x128); top head 256x3; all-but-one 896/2048/1152, end 126/107/128, push 23.6/7.4/22.7; all-32 end 0.0156x128 = 2; return 30-40% of first learning (384/1280=30%, 512/1280=40%); new-head share 0.815/0.986/0.937 = 82-99%; first audit 0.016/0.008/0.016 <= 2%.
