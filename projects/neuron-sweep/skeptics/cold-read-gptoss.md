# Cold read: "A local model could not make the split…" and "We then tried the largest model…"

Reader: ML researcher, not a mech-interp insider, no access to the discussion behind the post. I read the fold from "Unrelated jobs, or one job with offshoots?" down to get context.

## Paragraph 1: "A local model could not make the split into related and unrelated jobs."

1. **"A local model could not make the split into related and unrelated jobs."** (minor)
   Unclear: "local" could mean small, or it could mean running on our own machine. "The split" points back to the previous part, but that part has four labels, not two.
   Fix: "We also tried to get a model running on our own laptop to do this labelling. It could not tell related jobs from unrelated ones."

2. **"Before we used agent labellers, we tried Qwen3-14B at 4-bit."** (minor)
   Unclear: "4-bit" is never explained. Readers outside ML engineering won't know it means quantized weights.
   Fix: "Before we used the agent labellers, we tried Qwen3-14B with its weights compressed to 4 bits so it fits in memory."

3. **"(We did not run GPT-4.1-mini on this split.)"** (minor)
   Unclear: GPT-4.1-mini is the judge that worked earlier (82%, kappa 0.69). The reader asks why it was skipped, and the sentence doesn't say.
   Fix: Give the reason in one clause, e.g. "We did not try GPT-4.1-mini here, because …". Otherwise cut the sentence.

4. **"At that point, 259 runs had reference labels."** (BLOCKING)
   Unclear: The number doesn't match the text above. Above, the reference labels covered 84 + 100 = 184 runs for the judge, and the split used 200 runs. Where does 259 come from? It also doesn't say whether these 259 labels used the new four-label scheme (no effect / one job / unrelated / broken) or the old typical / one thing / several things scheme. If it's the old scheme, the kappa that follows is meaningless for "the split".
   Fix: "At that point, the first labeller had given 259 runs one of the four new labels. These included [say which runs]." If the 259 includes runs outside the 200, say so.

5. **"Qwen3-14B agreed with the reference labels on 65% of these runs (kappa 0.40)."** (minor)
   Unclear: The reader needs a yardstick. The two agent labellers agreed with each other at 79% (kappa 0.62).
   Fix: Add "The two agent labellers agreed with each other at kappa 0.62."

6. **"Next, we asked Qwen3-14B to first name the odd thing in each text."** (minor)
   Unclear: "Odd thing" is vague. It also assumes every text has something odd, which may be what caused the error described next.
   Fix: "Next, we asked Qwen3-14B to first say what was unusual in each text, and then give a label."

7. **"Qwen3-14B then called 12 of 15 no-effect runs "unrelated" (kappa 0.18 on 56 runs)."** (minor, borderline BLOCKING)
   Unclear: The denominator changes from 259 to 56 with no reason given. It also doesn't say whose labels count as "no-effect". Presumably the reference labels.
   Fix: "We tested this prompt on 56 runs. Of the 15 runs that the reference labels call no effect, Qwen3-14B called 12 'unrelated'. Its kappa fell to 0.18."

8. **"The reason: the three ordinary texts in a run are each about a different everyday topic."** (minor)
   Unclear: The point is clear once you work it out: unpushed texts naturally differ, so they look like "unrelated jobs". But "ordinary" is a new word for what the post elsewhere calls "typical" or "no effect".
   Fix: "Even with no push, the three texts of a run are about different everyday topics. Qwen3-14B read that difference as unrelated jobs."

## Paragraph 2: "We then tried the largest model that runs on our 16 GB machine"

9. **"It fills the machine's GPU memory limit (12.7 GB)."** (minor)
   Unclear: The machine has 16 GB, but the limit is 12.7 GB. Readers who don't know about Apple unified memory will see a contradiction.
   Fix: "Of the machine's 16 GB, the GPU may use 12.7 GB. gpt-oss-20b fills all of it."

10. **"Before the first run, we fixed a pass rule: …"** and **"We stopped that run at 68 runs, because it could no longer pass."** (BLOCKING)
    Unclear: Throughout the post, "run" means one push plus its three texts. Here it also means one labelling job, and "that run at 68 runs" uses both meanings in one phrase. A reader trying to stay precise trips over it.
    Fix: Call the labelling jobs "attempts". "Before the first attempt, we fixed a pass rule…" and "We stopped the second attempt after 68 runs."

11. **"…kappa of at least 0.55 against each labeller, and "unrelated" on at most 5% of the runs that both labellers call no effect or one job."** (minor)
    Unclear: It's dense but readable. The reader doesn't see why 0.55. The second condition is the actual false-alarm test, and saying that would help.
    Fix: Split it. "We fixed a pass rule before the first attempt. The model must reach kappa 0.55 against each labeller. That is close to the labellers' 0.62 with each other. It also must not cry 'unrelated' too often. Among runs that both labellers call no effect or one job, it may call at most 5% 'unrelated'."

12. **"With the Qwen3-14B prompt, it scored kappa 0.26 against each labeller on the 200 runs, and it called 42 of 175 such runs "unrelated" (24%)."** (BLOCKING)
    Unclear:
    - "The Qwen3-14B prompt": the paragraph above describes two Qwen prompts (answer at once, and name the odd thing first). Which one?
    - "Such runs" points back two sentences, into the middle of the pass rule. Most readers won't connect it to "runs that both labellers call no effect or one job".
    - The 175 appears with no definition. Kappa of exactly 0.26 against both labellers also looks like a typo. If it's true, say "0.26 against both".
    Fix: "First we gave it the Qwen3-14B prompt that names the odd thing first. On the 200 runs, its kappa was 0.26 against each labeller. Both labellers call 175 of the runs no effect or one job. gpt-oss-20b called 42 of those 175 'unrelated' (24%), far above the 5% limit."

13. **"With the labellers' own rubric and longer reasoning, it called 16 of the first 58 such runs "unrelated" (28%; kappa 0.10 and 0.14, and 9 of 68 answers ran past the length limit)."** (BLOCKING)
    Unclear:
    - Two denominators, 58 and 68, sit in one sentence with no link between them. The reader has to work out that 58 of the first 68 runs were the agreed no-effect/one-job kind.
    - "Such runs" again.
    - "Length limit" isn't defined. Is it the reasoning-token budget?
    - The reader can't tell how the 9 overlong answers were scored. Were they counted as wrong, or left out of the kappa?
    - It doesn't say which kappa belongs to which labeller.
    Fix: "Second, we gave it the labellers' own rubric and a larger reasoning budget. We stopped after 68 runs. Of those 68, both labellers call 58 no effect or one job. gpt-oss-20b called 16 of those 58 'unrelated' (28%). Its kappa was 0.10 against the first labeller and 0.14 against the second. In 9 of the 68 runs, it hit the token limit before giving an answer. We counted those as [wrong / dropped]."

14. **"We stopped that run at 68 runs, because it could no longer pass."** (minor, beyond the "run" problem in item 10)
    Unclear: The reader has to do the sum alone. 5% of 175 is about 9 runs, and 16 are already over that.
    Fix: "It had already called 16 runs 'unrelated'. The rule allowed at most 9 in all 175 (5%). So it could not pass, and we stopped."

15. **"Both models make the same mistake."** (minor)
    Unclear: "Both" could mean Qwen3-14B and gpt-oss-20b, or gpt-oss-20b and GPT-4.1-mini, which was just named.
    Fix: "Qwen3-14B and gpt-oss-20b make the same mistake."

16. **"They call a run "unrelated" when each text repeats a different word or has its own small error."** (minor; seems to contradict the text above)
    Unclear: Paragraph 1 gave a different reason for Qwen's mistake: ordinary texts on different everyday topics. Also, the labellers' rubric would call "each text broken its own way" broken, not unrelated. So this is a confusion between unrelated and broken. That is the same border where the labellers themselves disagreed.
    Fix: "Both models call a run 'unrelated' whenever its three texts differ. That covers ordinary texts on different topics. It also covers texts that each repeat a different word or have their own small error. The labellers call the first no effect and the second broken."

17. **"So the split above comes from the two agent labellers, not from a model judge."** (minor)
    Unclear: The agent labellers are also AI models, so "not from a model judge" looks self-contradictory. Also, the first labeller wrote the reference labels that the local models were scored against. The reader may worry the test was circular, but that's out of scope here.
    Fix: "So the split above comes from the two AI agent labellers, not from a single-call model judge like GPT-4.1-mini or the local models."

VERDICT: FIX

## Re-read

I re-read the rewritten text, from "We also tried to get a model running on our own machine" through "not from a model judge."

1. Fixed. "On our own machine" now says what "local" means.
2. Fixed. The text now says "weights compressed to 4 bits".
3. Fixed. The text gives the reason: "we wanted a judge that runs locally".
4. Fixed. 259 = 84 + 95 + 80, the sources are listed, and the labels are the four new ones.
5. Fixed. The labellers' kappa of 0.62 is given as the yardstick.
6. Fixed. The prompt now asks the model to "say what was unusual … then give a label".
7. Fixed. The text now says "We tested this prompt on 56 runs" and names the 15 no-effect runs.
8. Fixed. The explanation moved to the closing paragraph, so paragraph 1 no longer brings in "ordinary".
9. Fixed. The text now says "Of the machine's 16 GB, the GPU may use 12.7 GB".
10. Fixed. The text now uses "attempt", and "run" again means only one push and its texts.
11. Fixed. The rule is split into two sentences, and 0.55 is tied to the labellers' 0.62.
12. Fixed. The prompt is named (Qwen's second prompt) and 175 is defined. The same 0.26 against both labellers stays; I accept it as the real result.
13. Fixed. 58 is now a subset of 68, the kappas are matched to labellers, and the cap is defined (2,000 tokens) along with how capped answers were scored.
14. Fixed. The text now says "at most 8 in all 175, so it could not pass". 5% of 175 is 8.75, so 8 checks out.
15. Fixed. Both models are named.
16. Fixed. The mistake is now described as calling any run whose texts differ "unrelated". The labellers call those runs no effect or one job, and that matches paragraph 1.
17. Mostly fixed. "AI agent labellers" makes clear they are models too. "Not from a model judge" is still slightly loose (minor).

New sentences that could confuse a reader (all minor):
- "the 84 runs and 95 of the 100 runs from the two judge checks above". The reader wonders why 95 and not 100. Fix: add a reason in a few words, e.g. "(5 had no agreed label)".
- "because we wanted a judge that runs locally". The final split still uses AI agents, which presumably do not run locally. The reader may ask why running locally mattered. One clause would help, e.g. cost or reproducibility.
- "Of the 15 runs that the labels call no effect". "The labels" means the first labeller's labels. Fix: "that the first labeller calls no effect".
- "We scored those as no effect." The 9 capped answers out of 68 count as no effect, which pulls the kappa down. It does not change the pass/fail result, because the second attempt already failed on the "unrelated" count. One short sentence saying so would help.

No blocking items remain.

VERDICT: CONFIRMED
