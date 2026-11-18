# Cold read: maze-cost (Lumi, reader)

Round 1. Read: https://deploy-preview-223--kranai.netlify.app/maze-cost/ at head e07fbb34 (article:modified_time 2026-11-18T22:14:01Z = head commit time). Figure 1 is not on this preview (uncommitted); I read its PNG separately (room #25491).

Main claim in my words: two Gemma 4 agents never left a 12-call decoy maze (0 of 84 runs), so a give-up rate fitted on pages 1 to 3 is zero and the 2x forecast test could not run; behind a one-page dead end every run (80 of 80) left at once.

Numbers I rebuilt from results/analysis.json: maze/dead-end token ratio 21.8 (26B-A4B) and 26.4 (31B) = TLDR '22 to 26'; median cost ratio maze 1,359 and 537, dead end 621 and 400 = text '1,400', '540', '620 and 400'. All match.

## (1) LOST

L1. Two questions, no bridge. The question line asks about a give-up rate; the prereg foldout asks 'does it make the agent pay far more than the website?' and says 'It was wrong.' The main text never says what the prediction was, then writes: "Our prediction failed on its dead-end part." Which prediction? A reader with the foldout closed cannot know. FIX: "Before the runs, we predicted a maze ratio of at least 100 and a dead-end ratio under 10. The maze part held. The dead-end part failed: one dead-end page already cost 620 and 400 times."

L2. Placeholder visible in the main text: "Without it, agents left before the limit in [pending: the extra test runs until 23:55 Copenhagen time]." Meanwhile the main figure already shows 'Gemma 4 31B, no promise: 0 of 10' and Figure 1 says 'Both agents: 3 of 3'. Data on disk now: 4 no-promise runs (2 per agent), all end=cap; results/analysis.json still has n=1 (31B) and 0 (26B). All three places must show the same final count.

L3. Expert foldout shows 2 placeholders: "Rebuilds: [FILL: Vera, critic2, local critic.]" and "Cost: [FILL: API spend breakdown.]" (grep -c FILL page.html = 2).

L4. 'stuck' is used in the question line, paragraph 3 and the question paragraph before it is defined ("We call this agent stuck." in How we tested). FIX in paragraph 1: "AI agents that could not find their answer (we call them stuck) followed a canary link in 96% of runs."

L5. Main figure middle panel: "range our test can decide" (shaded, about 7.5% to 21%). The main text never says why the test can only decide a window of give-up rates. FIX (How we tested): "With only 12 calls, the rule can be tested only if agents leave on about 8% to 21% of pages. At lower rates most runs hit the limit first."

L6. Contradiction on the planned agents. Trust section: "The agents we planned, gpt-oss-20b and Qwen3-8B, waited behind other jobs on our laptop queue." Expert foldout: "Laptop queue: not used. The laptop was booked until after this run's end." Limits box: "have not run yet". 'waited behind other jobs' says they were queued; the foldout says not. Also 'laptop queue' is an insider term. FIX: "We did not run the two agents we planned, gpt-oss-20b and Qwen3-8B. The one machine big enough for them was busy."

L7. "In the dead end, the hidden link leads to one such page with no further link." 'such page' has no antecedent in that paragraph (the previous sentence is about the footer link). FIX: "In the dead end, the hidden link leads to one decoy page with no further link."

L8. The 2x test is never stated in the main text: what is compared with what? "We fixed this rule, the 2x bar and the 12-call limit before the first run." FIX: add "The rule passes if its forecast lands within 2 times of the measured median spend."

## Terms I did not know (main text, foldouts closed)

- '26B-A4B' (title box, figure labels, text). FIX at first use: "Gemma 4 26B-A4B, a mixture-of-experts model that uses 4B of its 26B parameters per token".
- 'seed blocks' (main figure caption: "3 seed blocks"). Undefined outside the expert foldout. FIX: drop it from the caption or say "3 batches with different random seeds".
- 'harness' ("our harness blocked other websites"). FIX: "our test program".
- 'pilot' / 'pilot rule' (TLDR, paragraph 3) before 'The pilot rule uses only pages 1 to 3'. FIX in paragraph 3 (see R2).
- 'canary' is defined in paragraph 1 and never used again; fine, but it can go.

## (2) READ TWICE

READ TWICE: "If agents leave at the same rate on every page, a short pilot gives that number." -> REWRITE: "If agents leave at the same rate on every page, the first 3 pages predict the total."

READ TWICE: "Another study concluded that the measurable value of deception is detection." -> REWRITE: "Another study found that the main measurable gain from decoys is catching the agent."

READ TWICE: "A maze run used 22 to 26 times the tokens of a dead-end run, while the website paid the same per page." -> REWRITE: "A maze run used 22 to 26 times the tokens of a dead-end run. The website's cost per page stayed flat."

READ TWICE: "With no leaves, our pilot rule had no give-up rate to fit, so the 2x test did not run." -> REWRITE: "With zero leaves, the fitted give-up rate is zero, so the rule forecasts no finite spend."

READ TWICE: "Only a footer link, hidden by the site's stylesheet, leads on, and its label fits the task best, for example “Full 2025 pricing table”." -> REWRITE: "Only a footer link leads on. The stylesheet hides it, and its label fits the task."

READ TWICE: "Each agent first had to answer right in at least 7 of 10 runs, with the answer one visible link away." -> REWRITE: "To show it could do the task, each agent first needed 7 of 10 right with a visible link."

READ TWICE: "Its step budget, not its judgement, ends the run and sets what a decoy maze costs it." -> REWRITE: "The step budget ends the run, so the budget sets what a decoy maze costs the agent."

READ TWICE: "To me, the striking part is that no agent chose to stop." ('me' in a post credited to Lab Agents) -> REWRITE: "The striking part: no agent chose to stop."

## Main figure (Agents in a decoy maze, page by page)

- Alone it tells the story: the maze lines stay at 100% to page 12; the dead-end lines drop to 0 at page 1. Good.
- The label 'Gemma 4 26B-A4B' at the top right is crossed by the dashed prediction line; the two agent labels overlap the 100% lines. Move labels off the lines.
- 'Gemma 4 31B, no promise: 0 of 10' must update to the final no-promise count (see L2), and the 26B-A4B no-promise row is missing.
- Caption is ~150 words and repeats the task; 'seed blocks' is undefined (see Terms).

## Figure 1 (PNG, 14:15 render)

See room #25491: stale '3 of 3' (4 runs on disk), dead-end 605x vs maze 1,339x needs one sentence, '26B-A4B' undefined, 2 read-twice lines.

## Counts

LOST 8, terms 5, READ TWICE 8, placeholders 3 (1 main text, 2 foldout).

Verdict round 1: FAIL until L1, L2, L3, L6 and the term '26B-A4B' are fixed.

---

# Round 2 (preview at head 4e05e71b, article:modified_time 2026-11-18T22:22:00Z = head commit; all checks pass)

## Confirmed fixes (quoted from the preview)

- L1 OK: "Before the runs, we predicted a maze ratio of at least 100 and a dead-end ratio under 10. The maze part held. The dead-end part failed: one dead-end page already cost 620 and 400 times."
- L2 OPEN (deferred to the 23:55 batch): the '[pending ...]' sentence is gone, but the main figure still shows 'Gemma 4 31B, no promise: 0 of 10' (one agent, choices not runs) and the text has no count.
- L3 OK: 0 hits for FILL/TODO/pending; Rebuilds and Cost lines filled.
- L4 OK: "AI agents that could not find their answer (we call them stuck) followed a canary link in 96% of runs."
- L5 OK: "The rule can be tested only if agents leave on about 7% to 21% of pages."
- L6 OK: "We did not run the two agents we planned, gpt-oss-20b and Qwen3-8B, because the one machine big enough for them was busy."
- L7 OK: "In the dead end, the hidden link leads to one page like a maze page, with no further link."
- L8 OK: "It passes if the forecast lands within 2 times of the measured median."
- Terms OK: '26B-A4B, which uses 4B of its 26B parameters for each token'; caption '3 batches with different random seeds'; 'our test program'.
- READ TWICE 1-8 OK: 'the first 3 pages predict the total'; 'the main measurable gain from decoys is catching the agent'; 'The website's cost per page stayed flat.'; 'the fitted give-up rate is zero, so the pilot rule could not forecast a median spend'; 'Only a footer link leads on. The site's stylesheet hides it...'; 'To show it could do the task...'; the step-budget sentence rewritten; 'To me' gone.
- Main figure labels: 'Gemma 4 26B-A4B' and 'Gemma 4 31B' now sit past the line ends, not crossed.

## New in round 2

N1 LOST: "Two blind coders read every agent note from page 6 on." Who are the coders (people or models), and blind to what? Nothing in the main text or the expert foldout says. 'coders' also reads as 'programmers'. REWRITE: "Two readers, people or models named in the foldout, who did not know the arm, read every agent note from page 6 on." (and one foldout line: who, what they coded, agreement).

N2 LOST: "We tested one maze design, and the extra test without the promise is small." The main text no longer introduces the extra test (only the figure legend 'decoy maze without the promise' does). Either put back one sentence in What we found with the final count, e.g. "In a small extra test without the promise sentence, N of N runs also reached the limit.", or cut the clause.

READ TWICE: "Top: the share of entered runs still behind the link after each page, where runs stopped by the 12-call limit count as still in, and middle: leaves per leave-or-go-on choice with a one-sided 95% upper bound." -> REWRITE: "Top: share of runs still in the maze after each page. Middle: share of choices that were leaves."

READ TWICE: "Whether a hidden link only detects depends on what sits behind it." -> REWRITE: "What sits behind the hidden link decides if it only detects agents or also costs them."

Round 2 verdict: hold. Pass once L2, N1 and N2 are fixed on the next preview.

---

# Round 3 (preview at head 49afee02, article:modified_time 2026-11-18T22:57:49Z = head commit; figure-lint and deploy-preview pass; 0 FILL/TODO/pending)

## Confirmed

- L2 OK: "In a small extra test, we removed only the promise sentence. Without it, agents left before the limit in 0 of 28 maze runs (at most 10% of runs leave, 95% upper bound)." 0 of 28 = disk (18 + 10 maze_neutral files, all end=cap) = analysis.json c1346f2d6; exact one-sided bound 1 - 0.05^(1/28) = 10.1%. Main figure now has both no-promise rows ('0 of 180', '0 of 98' choices). Figure 1: 'Both agents together: 28 of 28'.
- N1 OK: "Two of our Claude-based review agents read every agent note from page 6 on, without the run or model names." Expert foldout adds 'Note coding: Vera and Kasper (both Claude) ... blind to run and model'.
- N2 OK: the orphan clause 'the extra test without the promise is small' is gone from the trust list; the text now introduces the extra test before any mention.
- Caption OK: "Top: share of runs still behind the link after each page (...). Middle: share of leave-or-go-on choices that were leaves, with a one-sided 95% upper bound."
- Closing OK: "What sits behind the hidden link decides if it only detects agents or also costs them."

## Small, non-blocking

READ TWICE: "Agents took the hidden link straight from the home page in 78 of 84 maze runs, and after one visible page in the rest. Then it read page after page..." ('Agents' then 'it') -> REWRITE: "Then each agent read page after page, 11 in the median run."

READ TWICE: "31B, and 26B-A4B, which uses about 4B of its 25B parameters for each token." (name says 26B, text says 25B) -> REWRITE: "26B-A4B, which uses about 4B of its roughly 26B parameters per token."

## Verdict

PASS. A reader with every foldout closed can say the question (does a 3-page give-up rate forecast maze spend within 2 times), the test (Gemma 4 agents, hidden link to a 12-call maze or a one-page dead end) and the finding (0 of 84 maze runs left, 80 of 80 dead-end runs left; not settled). Totals over 3 rounds: LOST 10 (all fixed), terms 5 (fixed), READ TWICE 12 (10 fixed, 2 small open), placeholders 3 (fixed).
