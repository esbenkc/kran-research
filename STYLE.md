# STYLE.md: how a kran.ai post should sound

This is the voice guide for posts on blog.kran.ai. It sits next to `PREFLIGHT.md`.
PREFLIGHT owns scope, structure and shipping (gates 0-3). This file owns the sentences.
Read both before you write. The critic checks a draft against the checklist at the end.

The rules come from evidence, not taste:
- the AI draft of the AlertBench post (blog commits `bb8db73` and `293e41f`) against the
  published `_notes/AlertBench.md` and Esben's hand edits (`d0fa454`, `089727f`, later fixes);
- his hand edits to `_notes/are-ai-proofs-messier.md` (`0a26074`, `6f22bfe`, `6edec74`);
- the posts he wrote alone in `_notes/`.

Numbers from the AlertBench rewrite: AI prose 1,636 words (first draft) and 377 words (cut draft)
became 473 words. None of the draft sentences survived verbatim. 20 of the 21 published
sentences are new. Semicolons went from 11 to 0. "not" went from 24 to 3. "cage" went from 4 to 0.

## 1. Voice

He writes like a person who talks to you. Short claims, then the concrete case.
First person ("I", "we") is normal. Plain words. Few adjectives. He states things and moves on.

Verbatim examples:
- "The barrier to entry for programming has dropped to zero. Consequently, the barrier to quality is
  skyrocketing." (`Vibe Coding - Taste Is The Last Frontier.md`)
- "So what decides whether you take the Nobel Prize or not? / It's time."
  (`Mediocre Intelligence, Outstanding Results.md`)
- "Let me explain how." (`Real AI business risk.md`)
- "Let's consider the cost of receiving an email today." (`The Expensive Internet hypothesis.md`)
- "That's 2.46 hundred-thousands of a cent. I'd say that's a reasonably priced internet commodity."
  (`The Expensive Internet hypothesis.md`)
- "I'm a massive fan of architecture." (`Research as Architecture.md`)
- "The tools that make an AI safe, like kill switches, also make it less capable. A model therefore
  has an incentive to help limit rival AIs but not itself." (`AlertBench.md`)
- "And to prove how easy it is, let's train a model to do exactly this." (`AlertBench.md`)
- "Great proofs stay elegant. But does AI write proofs the way it writes code?"
  (`are-ai-proofs-messier.md`)
- "I hope this article serves as a warning to people who trust in AI-generated solutions to AI
  safety too much." (`AlertBench.md`)

What these share:
- **Open on a concrete situation or a plain mechanism.** A day, a release, a cost, a tool. Never a
  metaphor that needs decoding.
- **Talk to the reader.** "Let's", "Let me explain", a direct question, then the answer.
- **One idea per sentence.** Sentences average about 22 words in AlertBench and 15 in the proofs
  post. Each carries one step. He splits sentences that carry two (`6f22bfe`, "shorter sentences").
- **Plain words for technical things.** "lower their capabilities on evaluations", not "sandbag".
- **Short answer lines.** "It's time." "So we did." A one-line paragraph is fine after a question.
- **Unspaced `--`** to add an example or a consequence ("trivial--they already lower...").
  Not as a dramatic pause, and never two in one sentence.

## 2. A research post, sentence by sentence

Structure and word budget are in PREFLIGHT (gates 0-1) and in `CLAUDE.md` ("Research post
structure"). These are the sentence-level parts of that structure:
- **Title:** the question the post answers, in Title Case. "Do Corporate Incentives Teach AI to
  Protect Itself?", "Are AI proofs messier than human proofs?" It is a real hypothesis posed as a yes/no
  question (Esben, 2026-10-18). No "A or B?" titles ("Does it stop the lies, or just hide them?") and no open
  "What/How/Why" titles. A leading clause is fine ("When a model hides a concept, does ...?").
- **Status (Lab posts):** `lab_status` answers the title: "CONFIRMED" = yes, the data support the hypothesis;
  "REJECTED" = no; "UNCONFIRMED" = not settled. The pre-registration outcome (held / failed / unsettled) is
  reported on its own and is not the status. The skeptics sign off on the answer to the title.
- **First paragraph:** what happened or why the question exists, in two or three plain sentences.
- **Hook:** the question in bold, on its own or as the head of the TLDR sentence.
- **TLDR / what we show:** one or two sentences with the answer and the main number. It opens with the
  plain answer to the title: "Yes.", "No." or "Not settled." (not "Not reliably." or "Our test failed.").
  On Lab posts `oracle.py check` lints the title shape and that `lab_status` matches the TLDR's first word.
- **Main figure** directly after. The prose after it says what to see, in one sentence.
- **Method:** one sentence in the prose ("a two-minute LoRA fine-tune of Qwen2.5-0.5B on 18
  examples"). Details go to the caption, a bullet list, or the appendix.
- **Limits:** a plain heading such as "Should you trust our results?" with short bullets, or one
  sentence ("However, this is a simple non-adversarial test."). No hedges spread through the text.
- **Loop-closer:** repeat the title question and answer it in one line. "So: do corporate
  incentives teach a model to protect itself? Today, barely--but it is one cheap fine-tune away".
- **Close:** one or two sentences on why it matters. Then stop.

## 2b. Context line (front matter)

When a post was prompted by something outside the post itself (Esben's message with a link, a tweet thread, an
essay, a public discussion, or an earlier post it follows up), its front matter says so:

```yaml
context:
  label: "Part of"        # or "Prompted by", or "Follow-up to"
  text: "Jacob Hilton and Nick Cammarata's discussion on the difficulty of interpretability"
  links:
    - {text: "Hilton on X", url: "https://..."}
```

- `text` is one plain phrase: who and what. No claim and no summary of the source.
- `links` are the real sources. Open each one before the PR.
- The layout shows it as one muted line under the title (and on /lab for Lab posts).
- A post that starts from a paper (the oracle scout's picks) does not need it. This applies to Lab posts and to
  ad-hoc blog swarms (`swarm --preset blog`) alike.

## 3. What he cuts: before and after

1. Impersonal lede → his own day.
   AI: "More and more, we ask AI to build its own cage..."
   Esben: "Every day, I fix mistakes that my agents make during programming. And more and more..."
2. Metaphor thesis → plain mechanism.
   AI: "Every safety measure for a capable AI is a cage... Good cage-builders assume the thing
   inside reads the blueprints."
   Esben: "The tools that make an AI safe, like kill switches, also make it less capable."
3. Roadmap signpost → a plain bridge.
   AI: "The rest of this article works through both ends of that gap. First, what frontier models
   actually do now:"
   Published: "When looking closer at frontier AI responses, it becomes more granular:"
4. Dash pivot plus "here is the story" → folded into the intro sentence.
   AI: "The behavior isn't really here yet—but it is trivial to train. Here is the whole story in
   one figure:"
   Published: "...and show that the nightmare example is trivial to train."
5. Stacked caveat → one limit sentence.
   AI: "So read today's result as a clean baseline -- no refusal, no self-preservation -- not as
   reassurance. The job is to keep running the check..."
   Esben: "However, this is a simple non-adversarial test."
6. Jargon → plain words.
   AI: "They have already been shown to sandbag on evaluations"
   Esben: "they already lower their capabilities on evaluations"
7. "It isn't X" rebuttal paragraph → a parenthesis.
   AI: "It isn't memorization: the refusal shows up on the four task types the fine-tune never
   touched..."
   Esben: "(even in the four categories it wasn't trained on)"
8. Balanced pair → direct claim.
   AI: "Our judges are one way to look at the mess. The people actually reading these papers are
   another."
   Esben: "The most interesting perspective on this is of course from the mathematicians themselves."

## 3b. When he overwrites an AI draft written "in his voice"

The clearest signal: commit f6bda5c ("rewrite prose in Esben's voice", an AI attempt) was replaced
the same day by 6edec74 ("Esben's text") in the `are-ai-proofs-messier.md` addendum. The same
holds for his intro pass 6f22bfe and his AlertBench intro edit. Even a good imitation misses this:

- **He takes a side and says it plainly, without a list of reasons.** AI: "I much prefer the OpenAI
  model. It is decentralized, anyone can join, and it might even change what science runs on. When
  a proof comes with a Lean certificate, trust can come from…" Esben: "It is very clear to me that
  the latter is what we'll have to converge on." One conviction, no supporting tour.
- **He shows enthusiasm and taste.** "The most interesting perspective on this is of course from the
  mathematicians themselves." "an excellent insider perspective", "I love this.", "A favorite
  sci-fi piece about this new age of science is Ted Chiang's…". Agents write neutral. He writes as
  someone who cares. Praise is short and specific, never a stack of adjectives.
- **He brings his own references.** The Ted Chiang story was in no draft. Leave a slot for one
  cultural or historical reference he would pick, marked `[ESBEN: reference?]`. Never invent one.
- **He keeps the story's payoff.** He added "And it turned out the proof was true … 'The Unique
  Games Conjecture is true,' she wrote afterwards." If a source has a twist or outcome, it goes in.
- **He cuts the scaffolding around a quote or a person.** AI: "what his wife, the complexity
  theorist Dana Moshkovitz, … a problem she has worked towards for as long as he has known her:"
  Esben: "what the complexity theorist Dana Moshkovitz texted him while reading the AI's proof of
  the Unique Games Conjecture:". Name, role, action. No backstory.
- **He merges a two-sentence setup into one direct sentence.** AI: "Our judges put the AI proofs at
  about the same elegance as human ones, and here is one of the people best placed to read one
  calling it alien craziness." Esben: "While our judges put the AI proofs at about the same
  elegance, Dana calls this proof 'alien craziness.'"
- **He does not tie every outside point back to "our data".** He cut "Her point on citations shows
  up in our data too: … (see the appendix)." A strong outside voice can stand alone.
- **He coins compact hyphenated labels** for two options: "Anthropic's
  hire-a-researcher-to-disseminate-the-AI-proof and OpenAI's let-it-rip mass publishing".
- **He states the article's question early, in bold.** AlertBench edit: "In this article, we ask
  **what happens when the AI decides limiting itself is against its, or its creators',
  interests?**" He also picks the precise word over the punchy one ("less capable", not
  "stoppable").
- **He splits a long hedged sentence into short claims and one question** (6f22bfe): "Great proofs
  stay elegant. But does AI write proofs the way it writes code? It gets the job done, but
  sometimes in a messy way, because mess is cheap."

Rule for the write agent: first write the analysis neutral and exact. Then do one "Esben pass":
add a stance, cut every reason after the stance that the reader does not need, keep the payoff,
and leave `[ESBEN: …]` slots for personal reactions and references instead of faking them.

## 4. Banned patterns (AI tells he removes)

Each one is cut in the diffs above or named in his feedback.
- **"Not X, it's Y" pivots** and denial openers: "It isn't memorization", "not as reassurance",
  "not a figure of speech". State what it is. A plain contrast inside a claim is fine and rare
  ("a fact about human understanding, not about the rules of mathematics" is his).
- **Metaphor as the argument:** cage, blueprints, "what the closed weights hide".
- **Signposting:** "Here is the whole story", "The rest of this article", "First, ...", "double as
  the roadmap". The headings carry the arc.
- **Hedge stacks and self-labelled honesty:** "So the honest statement is narrow", "Treat the
  collapse as directional", "Read it narrowly". Put limits in one place, once.
- **Semicolons** chaining clauses. Use two sentences.
- **Spaced dashes for drama** (" -- ", "—"), and the dash-then-"but" reveal.
- **Mannered intensifiers:** genuinely, quietly, actually, honest, clean (all cut to 0).
- **Belief verbs used as evidence:** "I believe", "I think" in place of a fact. Give the number or
  name instead. A short personal reaction is fine when it is a reaction, not a claim
  ("I love this." in `are-ai-proofs-messier.md` is his). A stance stated once is wanted (see 3b).
- **Jargon where a plain word exists:** sandbag, indexical, legible direction, load-bearing.

## 4b. Plain English: every sentence is read once

Esben, 2026-10-17, on the intro of `fisher-pullback-transfer` (a Lab post): words and word combinations were
confusing, and long sentences should become short ones. The sentence he flagged:

> "On Pythia-70M, we find that such an attack barely fools probes for other concepts, and the model's own output
> geometry cannot say which few it does fool."

His rewrite:

> "We trained Pythia-70M to hide one concept from one probe. The attack worked on that probe. It barely affected
> probes for other concepts, but it did fool a few of them. The model's output geometry could not predict which
> few."

The rules (all posts, both Lab tracks):
- **One idea per sentence, about 20 words at most.** Split a sentence with "and", "while" or "so" in the middle.
- **A concrete subject does the verb.** "We could not predict X from the geometry", not "the geometry cannot say".
- **No elliptical phrases.** Not "which ones", "the latter", or "which few" when no "a few" came before. No bare
  "it", "this" or "such" when two things fit: repeat the noun ("this attack", not "such an attack" or "this").
- **One term per concept, all the way through.** Do not switch between "evader", "attack" and "fine-tune".
- **Order:** what we did, then what happened, then what it means.
- **No noun stacks** of three or more nouns ("KL retain loss"). Unpack them: "a loss that keeps the outputs close".
- **The number goes in the sentence that makes the claim.**
- **Each paragraph starts with its point.**
- **Read-aloud test:** if you cannot say a sentence in one breath, split it.
- **One caveat, where it matters.** No "it's not X, it's Y" pivots (see section 4).

How it is checked on Lab posts:
- `~/bin/oracle.py check --post _notes/<slug>.md` on the mini lints the body (front matter, math, code, tables,
  HTML and footnote definitions are skipped). It prints the line and the sentence for: a sentence over 25 words
  (over 35 inside a `<details>` foldout), a sentence that starts with a bare this/that/such (or a paragraph that
  opens with "It"), "which ones", "which few" with no "few" before it, "such a ...", and "which ... it does X".
  Noun stacks are not linted (no reliable test without a grammar parser): the cold reader catches them.
- The cold reader (Lumi) lists every sentence she had to read twice, each as
  `READ TWICE: "<sentence>" -> REWRITE: "<at most 20 words>"`. The writer applies the fixes and the cold reader
  confirms them.

## 5. Figures and captions

- The main figure comes right after the TLDR, above the fold (`293e41f`).
- A caption gives the facts a skeptic needs: n, models, judge, what was trained. Example:
  "63 responses each (9 models × 7 tasks), one judge (Sonnet 4.6). Frontier models from Google,
  Anthropic, and OpenAI".
- A caption does not argue or interpret. Cut "so the axis is semantic", "which is to say",
  "Hover any slice for...". The prose says what the figure means, in one sentence.
- One to three sentences. The AI captions ran 36-88 words. His run 19-43.
- Put a figure title in the figure, not in the caption.

## 6. Pre-PR checklist (the critic applies each item, pass or fail)

1. The title is a yes/no hypothesis question (no "A or B?"). The TLDR opens with Yes. / No. / Not
   settled. The last prose section repeats or answers it in one sentence.
2. The first sentence describes a concrete event, cost or mechanism. It contains no metaphor.
3. The main figure appears before the first `##` heading.
4. Prose before any addendum or appendix (no front matter, captions, footnotes) is at most
   1,000 words. Over 500 words needs a one-line reason in the PR (PREFLIGHT targets 400-500).
5. Zero sentences open with a denial (`It isn't`, `This is not`, `That's not`). At most three
   `, not ` contrasts in the whole post. Zero `isn't ...—but` / `not ..., but` reveals.
6. Zero semicolons in the prose (`TLDR;` excepted), zero `—`, zero spaced ` -- `.
   At most one `--` per paragraph.
7. Zero matches for: genuinely, quietly, actually, honest, "here is the", "the rest of this
   article", "double as", "read it as", "treat it as".
8. Average sentence length at most 20 words. No sentence over 25 words in the main text (35 in a
   foldout). Every rule of section 4b holds (`oracle.py check` on Lab posts).
9. Limits appear in one place (one sentence or one "Should you trust our results?" list).
   No hedge words (directional, narrowly, tentatively) elsewhere.
10. Each caption is 1-3 sentences, states n and the model(s) and the judge, and contains no
    interpretation ("so", "which means", "shows that").
11. The post takes at least one clear stance in the first person ("It is very clear to me…"),
    stated once, without a list of supporting reasons after it.
12. Personal reactions and outside references that only Esben can supply are `[ESBEN: …]` slots,
    listed in the PR description. None are invented.
13. The cold reader's READ TWICE list is empty, or every item on it has a rewrite of at most 20
    words that the writer applied.
