# Cold read: Would a Random Network Pass the 'One Main Job' Neuron Audit?

Preview read: https://deploy-preview-26--kranai.netlify.app/random-network-neuron-audit at 2026-10-15 15:55 PDT

Reader persona: competent ML researcher, not a mech-interp insider. One pass, top to bottom, banner → prediction box → body → figure → footnotes.

## Stumbling points

1. **"The Lab · Run and written by AI and rebuilt by three independent critics."** (banner)
   Why: "rebuilt" has no object. Rebuilt the analysis? The post? The code? I couldn't tell what the critics did.
   Fix: "Run and written by AI agents; three independent critics re-ran the analysis from the raw data."

2. **"Pre-registered prediction, posted in the swarm room at 23:16 Copenhagen time, before any data was collected"**
   Why: "swarm room" is internal jargon. It doesn't stop me, but it makes me wonder where this was posted and whether it's verifiable.
   Fix: "posted in our agents' shared log (timestamped) at 23:16…"

3. **"H1 (primary): non-lexical one-job rate A >= B+25pp, p_expected=0.70. H2: A >= C+15pp, p=0.70. H3: B any-type one-job >=40%, mostly lexical, p=0.55. H4: A any-type in [50%,90%], p=0.60."**
   Why: it's a block of code-like shorthand read before any concept is defined. "pp" isn't spelled out. "p_expected"/"p" reads as a p-value, not as subjective confidence. "any-type" isn't defined anywhere in the post. "one-job rate" isn't defined until the methods section.
   Fix: under the quote, add a plain-English line per hypothesis, e.g. "H1: among jobs that aren't a single word, trained neurons pass ≥25 points more often than random-network neurons (we gave this 70% odds)." Also define "any-type = counting all jobs, including single-word ones".

4. **"H3: B any-type one-job >=40%, mostly lexical, p=0.55."**
   Why: this seems to contradict the framing. The post is titled "would a random network pass?" and argues "an audit is evidence only if it can fail", yet the team predicted that ≥40% of random-network neurons *would* pass. The post never explains why (I'm guessing because of the Heap et al. result: random networks still have token-detector neurons). So the 0% outcome reads as both "the audit works" and "our prediction failed", and I can't tell which one the authors think matters.
   Fix: one sentence: "We expected random-weight neurons to often pass on single words, since random networks still react to specific tokens (Heap et al.); the real test was the non-lexical gap."

5. **"Before any judge call, GPT-4.1-mini replaced GPT-5-mini, which our account could not use."**
   Why: "judge" hasn't been introduced at this point, so this reads as a non sequitur.
   Fix: "Before any data was collected, we swapped one of the two LLM judges (which label neurons) from GPT-5-mini to GPT-4.1-mini, because our account couldn't access GPT-5-mini."

6. **"H1 and H2 missed by far (1.7 and 1.3 points)."**
   Why: I misread this. "Missed by 1.7 points" sounds like a near miss, which contradicts "by far". It only makes sense once you realise 1.7 and 1.3 are the *observed* gaps, set against predicted gaps of ≥25 and ≥15.
   Fix: "H1 and H2 missed by far: the non-lexical gaps were 1.7 and 1.3 points, against ≥25 and ≥15 predicted."

7. **"Random-weight neurons passed less often than predicted (0%), trained neurons far less (14%)."**
   Why: "than predicted" has no number next to it. I had to map this back to H3 (≥40%) and H4 (50–90%) myself.
   Fix: "Random-weight neurons passed 0% (predicted ≥40%); trained neurons 14% (predicted 50–90%)."

8. **"A neuron in a language model is one number in one layer."**
   Why: ambiguous. One number in which vector: the residual stream, the attention output, or the MLP hidden layer? The figure subtitle later says "MLP units", and "3,072 neurons in a trained layer" suggests the MLP intermediate layer, but I had to infer that.
   Fix: "A neuron is one of the 3,072 hidden units inside a transformer layer's MLP: one number per token."

9. **"His headline number: in Qwen3-32B, about 90% of neuron poles (the strong positive or the strong negative side) have one main job."**
   Why: "pole" is defined, but it's not clear that this audit looks at only one pole. That only comes out later, in "no … negative side". So I didn't know whether 14% and 90% are measured on the same kind of unit.
   Fix: add "We audit only the positive pole of each neuron."

10. **"TLDR; No. Not one of 150 random-weight neurons passes. The audit is strict, though: 14% of trained neurons pass (6% if both judges must agree), and 1.7% of random directions."**
    Why: "random directions" hasn't been defined yet (that happens in the next section). "6% if both judges must agree" implies the 14% is something else, but the post never says how two pass/fail verdicts become one 14% (mean of the two judges' pass rates? either judge?).
    Fix: "…and 1.7% of random directions (random mixes of trained neurons, a control). 14% is the average of the two judges' pass rates; 6% if both must agree."

11. **"We did not pre-register this comparison."**
    Why: unclear referent. Which comparison? The box says the random network (B) *was* in H1 and H3, so "we did not pre-register this" seems to contradict it.
    Fix: "Our pre-registered test compared non-single-word jobs; this all-jobs comparison of trained vs random-network neurons was not the pre-registered test."

12. **"One main job: at least 15 of 20 top examples fit one label, and the label passes the detection check. Light bars also count related offshoots, and hollow dots show layers 4, 14 and 24 alone."** (figure caption)
    Why: the caption sits above the methods, so "detection check" and "related offshoots" are both undefined when I first read it.
    Fix: "…and a second judge, given only the label, can pick out the unit's real examples among random ones (≥4 of 5). Light bars also accept small side jobs related to the main one."

13. **"Trained neurons pass most often at layer 4 (23%) and less often deeper in (10%)."**
    Why: I don't know how deep Qwen3-0.6B is, so I can't tell whether layer 24 is near the end.
    Fix: "…at layer 4 of 28 (23%) and less often at layers 14 and 24 (10%)."

14. **"The kinds are trained neurons, random-weight neurons, and random directions: fixed random mixes of all 3,072 neurons in a trained layer."**
    Why: the definition is fine, but the *purpose* isn't given until two paragraphs later. A reader without the "privileged basis" idea won't see why random directions are a control.
    Fix: append "If neurons are special units, they should be easier to read than arbitrary mixes of them."

15. **"The other judge saw only the label and 10 new excerpts, five of the unit's next-best and five random, and picked the five that fit."**
    Why: "the other judge" implies each judge labels and the other one checks, but the post doesn't say so outright. That matters for what "averaged" and "both judges must agree" mean.
    Fix: "Each judge labels every unit; the other judge then tests that label: …"

16. **"This fits Cammarata's claim that neurons, more than arbitrary directions, are the units to read, though by less than the 15 points we predicted."**
    Why: the arithmetic stalls me. 14 − 1.7 = 12.3, but the box says this gap is 13 points. (Probably rounding, e.g. 14.4 − 1.7.) Also "by less than 15 points" is the any-type gap, while H2 was about the non-lexical gap, so it's unclear which prediction this sentence checks.
    Fix: give the unrounded figures (e.g. "14.4% vs 1.7%, a 12.7-point gap; H2 predicted ≥15 points for non-single-word jobs, and that gap was 1.3").

17. **"If related offshoots (small side jobs) also count, a rule we fixed before the data but did not pre-register, 27% pass."**
    Why: "fixed before the data but did not pre-register" draws a distinction an outsider won't understand. If it was fixed in advance, how is that different from pre-registering it?
    Fix: "…a rule we wrote down before seeing data but left out of the public prediction…"

18. **"Our audit is simpler than Cammarata's: small judges, one pass, the strongest positive examples in 64-token windows, no steering, negative side or long context."**
    Why: "steering" is undefined (as a check, it means adding the neuron's activation and seeing what changes). "one pass" is ambiguous. "negative side" only makes sense if you remember "poles" from paragraph 3.
    Fix: "…no steering (pushing the neuron up and checking the model's output changes as the label predicts), no negative pole, and one labelling attempt per unit."

19. **"The 14% depends on our cut of 15 of 20. At 12 of 20 it is 31%, at 10 of 20 38%. The order (trained neurons, then random directions, then the random network) holds at every cut."**
    Why: number without context. The headline is "the random network fails", but I'm not told the random network's rate at the looser cuts. Does it stay at 0%, or rise to 10%? Whether the headline survives depends on it.
    Fix: give all three arms at each cut: "At 12 of 20: 31% / x% / y%; at 10 of 20: 38% / x% / y%."

20. **"On trained neurons, the judges agree on 83% of verdicts (Cohen's kappa 0.33)."**
    Why: these two numbers look contradictory: high agreement but weak kappa. The reason is that most units fail, so agreement by chance is high. The post doesn't say that, so I can't tell whether the judges are reliable.
    Fix: "…83% (mostly on failures; chance-corrected kappa is 0.33, i.e. they often disagree on which units pass)."

21. **"Sets of 20 random excerpts never passed (0 of 60), and nor did planted two-job units built from two neurons that each passed (0 of 17)."**
    Why: no stated purpose. I had to work out that these are a false-positive check and a "does the audit catch a two-job unit" check.
    Fix: lead with the purpose: "Two sanity checks: pure noise never passed (0 of 60), and the audit caught every planted two-job unit (0 of 17 passed)."

22. **"The random network's top examples fall early in a text more often: 11-20% within the first 16 tokens, against 5% by chance."**
    Why: no implication given. Is this a confound or a known random-network artifact, and does it bias the result? It ends the caveats list without telling me what to conclude. "5% by chance" also has no basis given (16 / ~320 tokens per document?).
    Fix: add the consequence: "so random-network neurons partly respond to position, not content, which the judges can't name; this makes the 0% a fair fail rather than a judge blind spot."

23. **"The audit tells learned structure from noise, in a contrast we did not pre-register as the test."**
    Why: same confusion as #11. The box lists the random network as the comparison in H1, so "did not pre-register" seems to contradict it.
    Fix: "…in the all-jobs comparison; our pre-registered test (non-single-word jobs) failed because almost no jobs were non-lexical."

24. **"How many trained neurons pass depends on the cut, so our rate cannot test Cammarata's 90%."**
    Why: this seems to contradict H4, which pre-registered exactly that test (50–90%) and is reported as "missed". Was H4 a meaningful test or not?
    Fix: "H4 missed, but because the rate depends on our cut and our audit is weaker than Cammarata's, this miss doesn't refute his 90%."

## Could I state the result?

After one read: the authors ran an LLM-judge audit (label a neuron from its 20 top-activating excerpts, then check the label on held-out examples) on Qwen3-0.6B. The audit passed 0% of neurons in a randomly initialized copy, 1.7% of random directions in the trained model and 14% of trained neurons, so the audit can tell a trained network from an untrained one. All four pre-registered hypotheses failed: the predicted gaps were for *non-single-word* jobs, and almost every passing job was a single word or token, so those gaps were tiny (1.7 / 1.3 points). The overall pass rates were also far below the predicted ranges.

Unsure about:
- Whether the "random network fails" headline is the pre-registered result or a post-hoc one. The box, the TLDR and the conclusion seem to say different things (#4, #11, #23).
- Whether 0% for the random network holds at looser cuts (#19).
- What "averaged" over two judges means for a pass/fail verdict (#10, #15).

## Figure

I could mostly read the main result off the figure (screenshot of the rendered SVG) plus its caption. There are three bars: trained neurons about 14% (dark) and 27% (light), random directions about 2% and 5%, and the random network at zero, explicitly labelled "0 of 150 units". That zero label is what makes the headline readable.

What confused me:
- The caption uses "detection check" and "related offshoots" before the text defines them (#12). The figure appears before the methods, so I couldn't tell what "pass" meant on first view.
- The light and dark bars overlap. The caption says light bars "also count" offshoots, which suggests the light bar is the total, not a stacked addition, but the overlap looks stacked at first glance. Each bar has its own whisker, and nothing says which whisker belongs to which bar.
- The hollow dots are per-layer values ("left to right" = 4, 14, 24), but they sit on the bar at different heights, and at first I read them as individual units.
- The figure shows nothing about the pre-registered test. H1/H2 are about non-lexical jobs, which aren't plotted, and there are no marks for predicted values. From the figure alone, I'd conclude the study succeeded, while the box says every hypothesis failed. A marker for the H3/H4 predicted ranges, or a non-lexical panel, would close that gap.
- The caption describes the rule and the setup but never states the takeaway ("no random-network neuron passes; trained neurons pass 14%").

## Confirm pass (2026-10-15 16:01 PDT)

Fresh fetch of https://deploy-preview-26--kranai.netlify.app/random-network-neuron-audit (cache-busted query and reload in a headless browser). I read the full text, both expanded "If you're curious" sections, and the rendered figure.

| # | Status | Evidence (new text) |
|---|--------|---------------------|
| 1 | WONTFIX-ACCEPTABLE | The banner is unchanged ("rebuilt by three independent critics"). It's site-wide, and it doesn't block understanding of this post. |
| 2 | FIXED | "posted in the agents' shared, timestamped log at 23:16 Copenhagen time" |
| 3 | FIXED | "In plain words: A is trained Qwen3-0.6B neurons … H1: on jobs other than one word or string (non-lexical), A passes at least 25 percentage points more often than B. … Each p is our prior." |
| 4 | PARTLY | "H3: at least 40% of B passes, mostly on single words." It still doesn't say *why* they expected random neurons to pass. The Heap paragraph a few lines later ("often score about as well in random transformers as in trained ones") lets me infer it. Non-blocking. |
| 5 | FIXED | "A unit passes if language-model judges find one main job. … Before any data, GPT-4.1-mini replaced GPT-5-mini as a judge" |
| 6 | FIXED | "The non-lexical gaps were 1.7 and 1.3 points, as the judges called nearly every job lexical." |
| 7 | FIXED | "Random-weight neurons passed 0% (predicted 40% or more), trained neurons 14% (predicted 50-90%)." |
| 8 | FIXED | "A neuron in a language model is one hidden unit in a layer's MLP block, one number per token." |
| 9 | FIXED | "We audit only the positive pole." |
| 10 | PARTLY | "on average over two judges, 14% of trained neurons pass (6% if both must agree), and 1.7% of random directions." The averaging is now clear. "Random directions" is still undefined in the TLDR (the box calls them "random mixes of trained neurons"), but the reader can map one to the other. Non-blocking. |
| 11 | FIXED | "This all-jobs comparison was not our pre-registered test." The box now makes clear the registered test was non-lexical. |
| 12 | FIXED | Caption: "a second judge, given only the label, picks the unit's held-out examples out of random ones (at least 4 of 5). Light bars also accept small related side jobs (offshoots)" |
| 13 | FIXED | "Trained neurons pass most often at layer 4 of 28 (23%) and less often at layers 14 and 24 (10%)." |
| 14 | FIXED | "If neurons are special units, they should be easier to read than such mixes." |
| 15 | FIXED | "each labelled every unit … The other judge then saw only the label … We report the mean of the two judges' pass rates." |
| 16 | FIXED | "Trained neurons pass 14.3% … random directions 1.7%" plus the box "Over all jobs, the gaps (14.3 and 12.7)". The arithmetic now checks out. |
| 17 | FIXED | Caption: "a rule written down before the data but not in the public prediction." |
| 18 | FIXED | "one labelling attempt by small judges, … no negative pole, no long context, and no steering (pushing the neuron up and checking the output)." |
| 19 | FIXED | "At 10 of 20, it is 38%, against 13% for random directions and 2% for the random network." The full table in "Does the 15-of-20 cut change the answer?" gives all arms at five cuts. |
| 20 | FIXED | "The judges agree on 83% of trained-neuron verdicts, mostly failures. Corrected for chance, agreement is weak (Cohen's kappa 0.33)." |
| 21 | PARTLY | Moved into the cut section: "Sets of 20 random excerpts, which share no job, never passed at our cut (0 of 60). Neither did planted two-job units…" "Which share no job" signals the noise control. The two-job control's purpose ("the audit rejects a unit with two jobs") is still left implicit, and it sits under a heading about the cut, where I wouldn't look for it. Non-blocking. |
| 22 | PARTLY | "(11-20% in the first 16 tokens, against 5% for uniform positions)." The 5% baseline is now clear, but the bullet still gives no consequence (confound? why the random network fails?). It ends the caveat list on a "so what?". Non-blocking. |
| 23 | FIXED | "Over all jobs, the audit tells learned structure from noise, though our pre-registered test failed." |
| 24 | FIXED | "Our trained-neuron rate depends on the cut and on a weaker audit, so it does not refute Cammarata's 90%." |

Figure: much improved. The legend now explains bar nesting ("Dark bar inside it"), whisker colours ("black for the dark bar, grey for the light bar"), and L4/L14/L24 dot labels. Dashed red lines mark the H4/H3 predictions ("Predicted: 50–90%", "Predicted: at least 40%"), so the failed predictions are visible at a glance. The non-lexical H1/H2 still aren't plotted, but the box covers them. The takeaway sits in the TLDR just before the figure; the sentence right after it is the per-layer one. Acceptable under house style.

### New stumbling points from the revision

1. **"Over all jobs, the gaps (14.3 and 12.7) still fall short."** Short of what? The 25/15 thresholds were stated only for non-lexical jobs, so "still fall short" has no explicit referent. Fix: "still fall short of the 25 and 15 points predicted." Non-blocking.

2. Minor: "Predicted: 50–90%" is drawn as a single dashed line at 50%, not as a band, so the 90% end isn't visible. Since 14% falls below either end, it doesn't change the reading. Non-blocking.

No new blocking items. After one read I can now state the result and the outcome of every pre-registered hypothesis without contradiction.

VERDICT: CONFIRMED
