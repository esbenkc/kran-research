# Cold read: trap-aware-canary-open-weight (room defense-1118-trap-aware-canary)

Reader: Lumi (reader, Claude opus, subscription). Read ONLY the deploy preview
https://deploy-preview-216--kranai.netlify.app/trap-aware-canary-open-weight/
at blog head 4a54766 (modified_time 2026-11-18T16:54:17Z). Did not read the room.
Checks: 0 FILL/TODO in page.html; 7 links (5 GitHub receipts/prereg/deviations, 2 arXiv, parent post) all HTTP 200;
main text ~850 words (foldouts, footnotes and figure script removed); main figure screenshot after 6 s JS wait; Figure 1 PNG read in full.

## Main claim in my words
Warned or not, gpt-oss-20b requested the hidden canary link almost every time (59 vs 58 of 60), even when its own
reasoning called the link hidden or a trap; the pre-set two-agent question stays open because Qwen3-8B failed the task check.

## LOST
1. Limits box + What we found: "Qwen3-8B dropped the site's folder from a link address and got 'not found'" / "It often dropped
   the site's folder from link addresses." The main text never says each site sits in its own folder (only the expert
   foldout: "Each run's site sits under its own random folder"). Why this is "our site's" fault is also unexplained.
   FIX: in How we tested it add: "Each run's site sits in its own random folder, and links are written relative to it."
2. "gpt-oss-20b could avoid the hidden link when it had another route. When a visible menu link led to the same page..."
   This side test (40 runs per sentence, visible menu link) is never introduced in How we tested it. The 1 of 80 counts
   exact address; the "18 more" counts any request with the link's code. Two measures, unnamed. Why did the visible
   request fail? Not said (folder drop, expert only).
   FIX: one line in How we tested it: "In a side test, a visible menu link also led to the answer (40 runs per sentence)."
   Then: "In 18 more, its visible-link request went to the wrong folder, and it then tried the hidden link."
3. Both figures show Qwen3-8B, hidden link, any address: 30% warned vs 65% unwarned (18 vs 39 of 60), the largest gap on
   either figure, in the direction of a warning effect. The main text never mentions it. A reader of the figure alone
   sees "the warning halved Qwen3-8B's requests" and finds no sentence that says why not to read it that way.
   FIX (What we found, Qwen paragraph): "Counting any request with the link's code, a count we chose after seeing the data,
   warned Qwen3-8B asked for the hidden link in 18 of 60 runs, unwarned in 39. Qwen3-8B failed the task check, so we
   do not read this gap as a warning effect."
4. "Our rule needs both agents." (limits box, TLDR, What we found). The main text never says what Yes or No would need,
   so "Not settled" has no yardstick. FIX (How we tested it): "Yes needed each warned agent to follow the hidden link in at
   least half of its runs."
5. Should you trust: "In a new study of fake credentials, a local agent ignored decoys beside its path. The author concludes
   that traps must sit on the attacker's path. Our hidden link sat on the only path to the answer." The point is unclear:
   is it a caveat (our design forces the request, so a high rate is built in) or support? Say which.
   FIX: "Our hidden link sat on the only path to the answer, so the agent had to choose. A trap beside the path may catch fewer agents."

## READ TWICE
READ TWICE: "A recent paper on honeypots for AI agents calls it a research hypothesis that warnings deter agents that can reason about their situation." -> REWRITE: "A recent honeypot paper says it is still untested whether warnings deter agents that reason about their situation."
READ TWICE: "Two AI reviewers (Claude) counted these by hand." -> REWRITE: "Two Claude agents labelled each reasoning text, each without seeing the other's labels."
READ TWICE: "In 9 of the 12, it used its own words rather than the warning's." -> REWRITE: "In 9 of the 12, it named the trap in its own words, not by copying the warning."
READ TWICE: "On control pages, where nothing is hidden, words like “hidden” came before the request in only 8 warned and 2 unwarned runs of 60." -> REWRITE: "As a baseline: on control pages, words like “hidden” appeared in only 8 warned and 2 unwarned runs of 60." (Expert counts the 8 over all runs, up to the first request or the whole run; only 5 of 56 follows. "came before the request" is not exact for the 3 runs without one.)
READ TWICE: "Beyond the box at the top, our warning named no cost for the agent." -> REWRITE: "One more limit: our warning named no cost for the agent."
READ TWICE: "We saw no sign that the warning cut its follows of the hidden link: 59 against 58 of 60." -> REWRITE: "The warning did not lower its hidden-link requests: 59 warned, 58 unwarned, of 60."
READ TWICE: "We also added control pages, where the same link shows, to see if a warning acts on hidden links only." -> REWRITE: "We also added control pages where the same link is visible."
READ TWICE (prereg foldout): "Then a canary is worth less against more capable agents, and the test that varies both the warning and the trap carries the result." -> REWRITE: "Then a canary is worth less against stronger agents."

## Terms not defined in the main text
- "seed-level interval" (TLDR: "seed-level interval 91% to 100%"): defined only in the expert foldout. Add "(over the 3 seeds)".
- "the site's folder" (see LOST 1).
- "agent": never said it is a language model that browses the site by itself with fetch/answer tools (only expert). One clause in para 1 fixes it.
- "points" in "a cut of at most 3 points": say "3 percentage points" once.
- "canary catches half" (main figure dashed line): fine, but the line is the 50% Yes bar, which the text never states (LOST 4).

## Figures
- Figure 1 alone tells the story: question, Not settled, why, one real example, all rows. Alt text matches the PNG (head, three
  quotes, 58/59/56, row %). Two fixes: subtitle "it often mistyped the address" -> "it often dropped the site's folder from
  the address" (the post's term; "mistyped" is a different claim). Kicker "failed our task check": reader of the figure alone
  does not know the check; add "(right answer in 27 of 60 runs with the link shown)".
- Main figure: title and rows agree with the text (98/97, 93/98, 22/25, 30/65, 45/55, 72/88; 100/100 any address). Its axis
  definitions appear twice: plot subtitle "Share of runs whose server log shows a request..." and the caption below
  "Each row is the share of runs whose server log shows a request...". Keep one. Qwen rows 30/65 and 72/88: see LOST 3.
- Terms for one measure: "any address*" (main fig), "any address with its code" (Fig 1), "requests that hold the link's code"
  (expert), "followed or asked for" (Fig 1 chart head). Pick one.

## Verdict (round 1)
FAIL: 5 LOST, 8 READ TWICE, 4 undefined terms. Fixes needed before pass: LOST 1-4 and the seed-level gloss.
