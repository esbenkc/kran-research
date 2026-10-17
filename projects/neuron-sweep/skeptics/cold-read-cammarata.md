# Cold read: "What happens if you push a neuron up?" — new part (Cammarata split)

Source read: https://deploy-preview-39--kranai.netlify.app/random-network-neuron-audit (fold under "If you're curious"), checked against https://already-clean.goodfire.com/essay.md.

## 1. Where a non-insider ML researcher gets lost

1. **"It includes texts that are each broken in their own way, so it is no measure of superposition."**
   Why lost: the fold never said that "several things" was supposed to measure superposition, so the reader cannot tell what is being ruled out.
   Fix: "We had hoped that 'several things' would flag neurons that write unrelated things, which is what superposition predicts. It does not: it also catches runs where each text is broken in its own way."

2. **"only 5 of the judge’s 12 “several” calls matched our hand label."**
   Why lost: we are not told what the hand label said for the other 7 (one job? broken? no effect?), and that is the reason the label is called "loose".
   Fix: "...matched our hand label; we labelled the other 7 as one job (n), or as texts broken in different ways (m)." Also replace "held out from writing the prompt" with "that we did not use while writing the prompt".

3. **"The next part replaces it with Cammarata’s own test."**
   Why lost: Cammarata gives a definition, not a test. "Test" makes the reader look in the essay for a procedure that is not there. "It" is also ambiguous.
   Fix: "The next part replaces the 'several things' label with Cammarata's definition of a polysemantic neuron."

4. **"Cammarata calls a neuron polysemantic when it “appears to have multiple unrelated jobs”."** (with the caption's "by Cammarata's sense of polysemantic")
   Why lost: the essay's sentence goes on, "or that looks impossible to understand", and it gives this as the field's "broad name", not as his own definition. The post then puts "broken (garbled, with nothing you can name)" in a class of its own and never reports it. A reader who checks the essay sees half the definition dropped, and the dropped half is exactly the "broken" class. **Blocking** (see section 2).
   Fix: quote the whole clause and say how "broken" maps onto it. For example: "The essay calls a neuron polysemantic when it 'appears to have multiple unrelated jobs or that looks impossible to understand'. We count only the first half as polysemantic. Runs where the text is garbled we label 'broken' and report separately: x and y of 100 neuron runs, against z and w of random directions." Then soften the caption to "a split into one job, unrelated jobs and broken, following Cammarata's definition".

5. **"So we relabelled 200 runs at 10 times max, 100 neuron pushes and 100 random-direction pushes"**
   Why lost: we are not told why only 10 times (presumably because at max almost nothing changes), or how the 200 runs were chosen (at random from all runs, or only from runs the judge called changed?).
   Fix: "We used only the 10-times runs, because at max about 9 in 10 runs show no effect. We drew 100 neuron runs and 100 random-direction runs at random from all 10-times runs."

6. **"Two labellers worked blind to the kind of push: one of us, and a fresh agent that saw only the rubric and the texts."** together with Limits: **"hand labels from one of us"** and **"two labellers, both agents on our team."**
   Why lost: "one of us" and "hand labels" make a reader picture a human annotator. Only the last line of Limits says both labellers are agents. The reader has to revise what they trusted, and "hand labels" (also used for the 84 and 100 runs earlier) now reads as wrong. **Blocking.**
   Fix: say it the first time it comes up: "Two AI agents labelled the runs, both blind to the kind of push: the agent that wrote our earlier reference labels, and a fresh agent that saw only the rubric and the texts." Replace "hand label(s)" throughout with "reference labels (written by an agent on our team)". If any labels really were written by a human, say which ones.

7. **"Most disagreements were unrelated against broken, or no effect against one job."**
   Why lost: "X against Y" reads as a comparison of two results, not as "one labeller said X, the other said Y". It is also the most important caveat in the paragraph, because unrelated against broken is exactly the line being measured, and it gets no comment.
   Fix: "When they disagreed, it was mostly because one said 'unrelated' and the other 'broken', or one said 'no effect' and the other 'one job'. So the unrelated count is soft at its edge with broken."

8. **"Of the runs that change the text, 84% of neuron runs show one job, for both labellers. Random directions: 89%."**
   Why lost: no denominator (how many of the 100 runs changed the text, and by whose call: the judge's or the labellers'?). "For both labellers" can mean either "each labeller separately got 84%" or "both labellers agreed on 84%".
   Fix: "Each labeller marked about N of the 100 neuron runs as changed (anything but 'no effect'). Of those, each called 84% one job. For random directions: about M changed, 89% one job."

9. **"5 and 6 of 100 neuron runs (the two labellers), against 0 and 1 of 100 random-direction runs."** / **"(Fisher’s exact test, p = 0.06 and 0.12)"**
   Why lost: which count goes with which is only implied. The p-values check out only for labeller A 5 vs 0 (p = 0.059) and labeller B 6 vs 1 (p = 0.118). Paired the other way, 6 vs 0 gives p = 0.029, which is significant. The reader cannot rule that out. Also unstated: the test is two-sided (one-sided 5 vs 0 gives p = 0.03).
   Fix: "Labeller A: 5 of 100 neuron runs against 0 of 100 random-direction runs. Labeller B: 6 against 1. Not significant with 100 runs per arm (two-sided Fisher's exact test, p = 0.06 and 0.12)."

10. **"On the write side, unrelated jobs are rare, as Cammarata expects."**
    Why lost: Cammarata's numbers are about what neurons read (top activations). He makes no claim about steering at 10x max. Two paragraphs later the post says this steering "says nothing for or against Cammarata's 90%", which pulls the other way.
    Fix: "On the write side, unrelated jobs are rare. That fits Cammarata's picture, but he made no claim about steering at this strength."

11. **"so this test cannot give the credit to neurons."**
    Why lost: "give the credit" is a loose idiom. The reader has to work out that it means "cannot show that neurons are special units".
    Fix: "so this test cannot show that neurons write more cleanly than arbitrary directions."

12. **"A push this large mostly drives the model into one loop, whatever was pushed."**
    Why lost: this disagrees with the post's own numbers. The 10-times bullet says repetition (zlib) shows up in 46% of neuron one-thing runs and 41% for random directions. Since one thing is 55% of neuron runs, that is about a quarter of all neuron runs. That is not "mostly". **Blocking.**
    Fix: "A push this large often drives the model into repeating one token: about half of the one-job runs, for neurons and random directions alike." Or give a loop rate measured on the 200 relabelled runs.

13. **"We also tried a larger local judge for this split: Qwen3-14B, 4-bit, about 11 GB at peak, the largest model that fits on our 16 GB machine next to other jobs."**
    Why lost: the obvious question is why the GPT-4.1-mini judge used for the rest of the fold was not used for this split, and the post does not answer it. The memory details are noise to this reader.
    Fix: begin with the reason, e.g. "Before using agent labellers, we tried to automate the split with a local model, Qwen3-14B (the largest that fits our 16 GB machine)." Say in one clause why GPT-4.1-mini was not used.

14. **"Read from its first answer token, it gave our label in 65% of runs (kappa 0.40)"**
    Why lost: "read from its first answer token" is implementation jargon, and "65% of runs" has no denominator (the next sentence gives 56 runs for a different setup). "Our label" means whose label?
    Fix: "When made to answer with a label straight away, with no reasoning, it agreed with our reference labels on 65% of N runs (kappa 0.40)."

15. **"found none of the 5 unrelated runs we had labelled at that point."**
    Why lost: "5" also appears just above as one labeller's unrelated count out of 100 neuron runs. The reader takes these to be the same 5 runs, and "at that point" suggests they are not.
    Fix: "found none of the 5 runs we had labelled 'unrelated' so far (an earlier, smaller batch than the 200 above)."

16. **"it called 12 of 15 typical sets “unrelated”"**
    Why lost: "typical sets" mixes vocabularies. "Typical" was the GPT judge's label, the new scheme calls it "no effect", and "sets" has not been used before (elsewhere it is "runs" or "the three texts of a run").
    Fix: "it called 12 of the 15 no-effect runs 'unrelated', because the three ordinary texts in a run are each about a different everyday topic."

17. **Limits: "the positive pole only"**
    Why lost: "pole" was defined for reading (strongest positive or negative activations). For steering, the reader has to guess that it means "we only pushed neurons up, never down".
    Fix: "only upward pushes (no negative pushes)".

18. **Limits: "with the skeptics’ rebuilds."**
    Why lost: no "skeptics" have been mentioned anywhere in the post, so the reader does not know who rebuilt what.
    Fix: "with independent re-implementations of the analysis by reviewer agents", or drop the phrase.

19. **Caption: "The last panel relabels 200 runs at 10 times by Cammarata's sense of polysemantic, by two blind labellers."**
    Why lost: "relabels... by... by" is hard to parse. "Cammarata's sense" carries the half-quote problem from item 4, and "blind labellers" hides that they are agents (item 6).
    Fix: "Last panel: 200 runs at 10 times (100 neuron, 100 random-direction), labelled no effect / one job / unrelated jobs / broken by two AI agents blind to the kind of push."

20. **"One neuron run, for example, writes “Five”, “五” and “quint-“."**
    Why lost: the closing quote after "quint-" is an opening quote (“). Minor typo, but on a quoted token it makes the reader wonder whether the token includes the quote mark.
    Fix: use a closing quote: "quint-".

## 2. Fairness to Cammarata

- **"appears to have multiple unrelated jobs": partial, so NO as it stands.** The essay's sentence is: "We had canonical examples of polysemantic neurons, the broad name for a neuron that appears to have multiple unrelated jobs or that looks impossible to understand." Two problems. (a) The post presents this as "Cammarata calls a neuron polysemantic when…", but the essay gives it as the field's broad name. (b) The post cuts "or that looks impossible to understand" and then files garbled output separately as "broken". Under the full definition, "broken" runs would plausibly count as polysemantic. The words quoted are verbatim. The caption's "Cammarata's sense of polysemantic" overstates how faithful the post is (item 4).
- **"iy" → "diy" → "do it yourself": YES.** The essay says: "For instance, one neuron that detected "iy" grew to detect "diy," and then "do it yourself," which doesn't contain "iy" at all." (also in the Recursive Generalization section: "a language-model neuron responds to "iy," including in "Miyazaki," then to "diy," and then to "do it yourself""). Treating related offshoots as one job also matches the essay: "roughly 90% of neuron poles have one main job, often with small related offshoots".
- "as Cammarata expects" (item 10): this is an inference. The essay makes no steering prediction at 10x max. It mentions steering only as a confirmation check ("check its steering effects … which line up closely").

## 3. Internal consistency of the numbers

- Fractions in Limits: 512/3,072 = 1/6 and 256/3,072 = 1/12. Also 28 × 3,072 = 86,016, 28 × 512 = 14,336, 28 × 256 = 7,168, and 2,688 = 96 per layer. **Consistent.**
- Unrelated counts: 5 and 6 of 100 (neurons), 0 and 1 of 100 (random directions). Agreement on 3 neuron runs (≤ min(5,6)) and on 0 random-direction runs. **Consistent.**
- Fisher p-values: two-sided 5/100 vs 0/100 gives p = 0.0594 and 6/100 vs 1/100 gives p = 0.1184, which **match** "p = 0.06 and 0.12". Calling them "not significant" is correct at 0.05 two-sided. But the pairing is implicit: 6 vs 0 would give p = 0.029 (item 9). One-sided 5 vs 0 gives 0.030; say it was two-sided.
- 84% / 89% one job among changed runs: **cannot be checked**, because no denominator is given (item 8). It is plausible against the judge's 68% / 71% changed at 10x. For example, 57/68 = 84% leaves about 11 non-one-job runs, of which 5–6 are unrelated and so about 5–6 are broken. Exactly 84% for both labellers while they agree on only 79% of labels is possible but unusual; give the counts.
- "A push this large mostly drives the model into one loop": **inconsistent** with "In 46% of the neurons' one-thing runs (41% for random directions)… compress … better than 99% of no-push texts", which is about 25% of all neuron runs (item 12).
- "just as often" (one job, neurons 84% vs random 89%): consistent. If anything, random directions are higher.
- Judge "several" at 10x = 13% of neuron runs, against labellers' "unrelated" 5–6%. This fits "only 5 of the judge's 12 'several' calls matched", so it is **consistent** with the claim that "several" is loose.
- Agreement: 79% with kappa 0.62 over 4 classes is plausible. Qwen3-14B: 65% / kappa 0.40, and a kappa of 0.18 on 56 runs, given without the first denominator (item 14).
- Two different "5"s (5 of the judge's 12; 5 unrelated runs "at that point"; 5 of 100 for labeller A) are easy to mix up (items 2, 15).

VERDICT: FIX (4, 6, 12)

## Confirmation

Re-read of the rebuilt preview (same fold and section).

1. Fixed: there is now a bridge sentence, "We had hoped that “several things” would flag pushes that make the model write unrelated things, which is what superposition predicts."
2. Fixed: the post now says what the other 7 were ("no effect (5) or one thing (2)"; 5 + 2 = 7), and "held out" is reworded.
3. Fixed: "replaces this label with Cammarata’s split" (but see new item A).
4. Fixed (blocking): the post quotes the full sentence, "the broad name for a neuron that appears to have multiple unrelated jobs or that looks impossible to understand", maps the second half to "broken", and reports broken counts. The added quote "neurons learning unrelated things together" is verbatim from the essay ("say exactly what we mean: neurons learning unrelated things together because they rarely interfere").
5. Fixed: the 200 runs are now "drawn at random from all runs at 10 times max", with the reason for skipping max given.
6. Fixed (blocking): the post now says "Two AI agents labelled the runs", calls the earlier labels "reference labels" written by an agent, and Limits says "reference labels written by agents on our team".
7. Fixed: the post now says "one said “unrelated” and the other “broken”", plus the softness caveat.
8. Fixed: denominators given (74 and 67 changed for neurons; 71 and 62 for random directions).
9. Fixed: the pairing is explicit (first labeller 5 vs 0, second 6 vs 1) and the test is stated as two-sided.
10. Fixed: the post now says "the essay makes no claim about steering this hard."
11. Fixed: the post now says "cannot show that neurons write more cleanly than arbitrary directions."
12. Fixed (blocking): the post now says "often drives the model into repeating one token: in about half of the one-thing runs". This matches the earlier 46% and 41%.
13. Fixed: the reason is given ("so that it would not rest on a paid API"), and "We did not run GPT-4.1-mini on this split."
14. Fixed: the post now says "Made to answer with a label at once, with no reasoning", with a denominator of 259 runs (but see new item B).
15. Fixed: the 5 is now scoped as "the 5 runs among them labelled “unrelated”", so it no longer collides with the 5 of 100.
16. Fixed: the post now says "12 of 15 no-effect runs", with the reason in plain words.
17. Fixed: the post now says "upward pushes only".
18. Fixed: the post now says "independent rebuilds of the numbers by reviewer agents".
19. Fixed: the caption now names the four labels and "two AI agents, both blind to the kind of push".
20. Not fixed: the page still has “quint-“, with an opening quote where a closing quote belongs. Cosmetic, non-blocking.

New places where a reader gets lost (all non-blocking):

A. "The next part replaces this label with Cammarata’s split." The essay gives a definition, not a split. Suggest "with a split based on the essay's definition".
B. "it agreed with them on 65% of 259 runs". The fold so far names 84 and 100 reference-labelled runs (184 in all) and 200 agent-labelled runs, so the reader cannot tell where 259 comes from. Say which runs these were, e.g. "259 runs we had labelled by then (the 184 reference runs plus 75 of the 200)", whichever is true.
C. "in about half of the one-thing runs, for neurons and random directions alike". This sits in a bullet about the labellers' "one job" counts, but the 46% and 41% come from the GPT judge's "one thing" runs. Suggest "in about half of the judge's one-thing runs".
D. "so that it would not rest on a paid API". The split then rests on two AI agents, which a reader may assume also run on a paid API. Optional: drop the reason or say "on a closed model judge".

Number check of the new figures:
- One job: 62/74 = 83.8%, 56/67 = 83.6%, 63/71 = 88.7% and 55/62 = 88.7%. These round to the stated 84% and 89%.
- Changed = one job + unrelated + broken, per labeller:
  - Neurons: 74 = 62 + 5 + 7 and 67 = 56 + 6 + 5.
  - Random directions: 71 = 63 + 0 + 8 and 62 = 55 + 1 + 6.
  - All four add up (assuming the "Broken" bullet lists the first labeller first).
- Fisher p-values (0.06 and 0.12, two-sided) match the stated pairing.
- The judge's "several" calls: 5 + 5 + 2 = 12. **Consistent.**

VERDICT: CONFIRMED
