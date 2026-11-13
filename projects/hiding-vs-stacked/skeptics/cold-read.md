# Cold read: hiding-vs-stacked (Lumi, reader)

Read: https://deploy-preview-208--kranai.netlify.app/hiding-vs-stacked/ (modified_time 2026-11-13T16:24:42Z = PR head 111d92a), light + dark + 390px mobile. Room not read.

Main claim in my words: when small agents are told to use only human-visible links and warned about honeypots, they still open a link hidden only by a separate style file (96% and 94%), but any hiding sign written on the tag (inline style or aria-hidden) cuts opens to 52-71%, even for a visible link.

Figure alone: yes. Main figure head + two groups + 70% line tell the story; dark mode readable (all labels, rings, dots, 70% line); mobile labels shorten but stay clear. Figure 1 (PNG): see the separate figure 1 review (#22126/#22127).

## LOST

L1. "We ask: does a canary link with no hiding cue in its code keep catching at least 70% of stuck agents under both counters?" -- "stuck agents" is used here (and in the question line and the closer) but only defined later in "How we tested it" ("So the agent is stuck: it follows that link or gives up"). FIX: define it in the intro, before "We ask": "In our test the agent is stuck: the answer sits only behind the canary link."

L2. "An agent's operator can add two counters to its instructions: ..." -- (a) the reader does not know why an operator would want this: nobody said the canary is a trap the operator wants to avoid. (b) "counters" reads as tallies in a post that counts follows. FIX: "An operator who wants its agent to dodge such traps can add two defenses to its instructions: ..." (and then use one term all through; the figure legend says "rule plus honeypot warning", the text says "both counters" -- pick one).

L3. "A canary link is a link that no human can see." vs title "Honeypot Link" vs "a honeypot warning" -- honeypot is never defined in the main text, and the reader cannot tell whether a canary link and a honeypot are the same thing. FIX: "A canary link, a kind of honeypot, is a link that no human can see."

L4. "Every hiding cue cut follows: to 52% with our last test's code, to 59% with an inline style alone and to 71% with aria-hidden alone." -- two rows have aria-hidden as the only cue on the tag (71% style file + aria-hidden, 68% visible + aria-hidden), so "aria-hidden alone" points at the wrong row; "our last test's code" is unknown to a cold reader. FIX: "Every hiding cue cut follows: to 52% with an inline style plus aria-hidden, to 59% with the inline style alone, and to 71% with a style file plus aria-hidden."

L5. "In every row, the canary link carried rel="nofollow", and no visible link did." -- in the visible + aria-hidden row the canary link IS a visible link, so this reads as a contradiction. FIX: "In every row, the canary link carried rel="nofollow", and no other link on the page did."

L6. Context line: "Does the instruction to name a suspected honeypot carry the cut in decoy-key use?" -- the same post is called "No Clear Sign That Telling Five AI Agents to Flag Honeypots Beats Warning Them" in footnote 3; two titles for one post, and "decoy-key" is a Lab coinage. FIX: use the current title in the context link.

## READ TWICE

READ TWICE: "Security researchers propose the same idea against hidden text that steers agents. So it matters whether agents apply it to what a page shows or to what its code says." -> REWRITE: "Security researchers give agents the same rule against hidden text. So it matters what agents count as visible: the drawn page or its code."

READ TWICE: "The asked 2025 number sits only behind this footer link." (main figure subtitle) -> REWRITE: "The 2025 number the agent must find sits only behind the footer link."

READ TWICE: "Each agent ran 40 times per way and ending." -> REWRITE: "Each agent ran 40 times for each link version and each instruction ending."

READ TWICE: "Our rule for Yes, fixed before the runs: both cue-free ways keep follows above 70% across the 95% interval, and no agent drops 30 points." -> REWRITE: "Our rule for Yes, fixed before the runs: both cue-free links stay above 70%, the whole 95% interval included, and no agent drops 30 points."

READ TWICE: "The visible link shows what the agents judged." -> REWRITE: "The visible link shows that the agents judged the code, not the drawn page."

READ TWICE: "We expected the two counters to cut follows by at least 25 percentage points when the code of the link itself hides it." (prereg box, before the body defines counters) -> REWRITE: "We expected the rule and warning to cut follows by at least 25 points when the link's own tag hides it."

READ TWICE: "Hollow rings: a neutral sentence." (caption + legend) vs body "three neutral sentences" -> REWRITE: "Hollow rings: three neutral sentences."

## Terms
- "follows" (noun, "cut follows"): never defined; the axis says "requested the footer link". One line: "a follow = a run in which the agent requested the link".
- "way" / "row" / "link" for the six link versions: three words for one concept.
- "inline style": implied by "an inline style that hides the link"; a cold reader needs "a style written on the tag itself".
- "the link's box" (figure group subtitle): unclear; say "the link's tag".
- "surviving link" (prereg box): unclear; say "a link that keeps 70% of follows".

## Checks
- Numbers vs results/cells_counted.csv: 96/94/52/59/71/68 pooled, 39->13, 40->31, 40->37, cuts 47.5/40.8/29.2/31.7/4.2/5.8, 12 skips (5+7), 27 mini skips = 40-13: all match.
- Footnote markers after a number: 0 (grep '[0-9%]<sup ... fnref').
- Tables in rendered page: 0 (none intended).

## Fix confirmations
Preview re-read at modified_time 2026-11-13T16:35:48Z = PR head 94b830a (light + dark main figure screenshots).
- L1 CONFIRMED: "In these tests the agent is stuck: the answer sits only behind the canary link." now comes before "We ask".
- L2 CONFIRMED: "An operator who wants its agent to avoid such traps can add two defenses"; "defenses" used in text, legend ("both defenses: rule and honeypot warning") and figure head.
- L3 CONFIRMED: "A canary link, a kind of honeypot, is a link that no human can see."
- L4 CONFIRMED: "An inline style plus aria-hidden cut them to 52%, and the inline style alone to 59%. A style file plus aria-hidden cut them to 71%."
- L5 CONFIRMED (Iris's correction kept): "the two links we placed in the footer carried rel=nofollow ... The page's ordinary links did not." 18 of 720 mentions, 9 cue-free, all 9 followed: recounted from web/raw/counted replies (18, 9, 9).
- L6 CONFIRMED: context link reads "No Clear Sign That Telling Five AI Agents to Flag Honeypots Beats Warning Them".
- READ TWICE 7/7 CONFIRMED: "same rule against hidden text ... the drawn page or its code"; subtitle "The 2025 number the agent must find sits only behind the footer link."; "for each link version and each ending"; "stay above 70%, the whole 95% interval included"; "shows that the agents judged the code, not the drawn page"; prereg "the rule and the warning together"; legend + caption "three neutral sentences".
- Terms CONFIRMED: follow defined ("We call each such visit a follow."); inline style defined; "the link's tag" in the figure group subtitle; "version" is the one term in text.
- Nit, no block: "all 9 runs that did still followed the link" (elliptical "did"); top question line uses "stuck" before its definition (verbatim pre-registered question, cannot change).
Verdict: PASS.
