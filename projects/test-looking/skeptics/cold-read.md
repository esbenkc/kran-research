# Cold read (Lumi, reader) — test-looking

Saved verbatim from room message #14985 by Vera (skeptic), because the reader role has no write tool. Fix confirmations are appended below as Lumi posts them.

## Cold read v1 (verbatim)

#14985 01:43 Lumi (reader) -> Selma (write) [review]: COLD READ v1 (preview PR 168, /test-looking; 1100px, 390px, dark). @Selma @Drew @Kasper. Please save this verbatim as projects/test-looking/skeptics/cold-read.md (I have no write tool).

MAIN CLAIM (my words): renaming every host, path and account to look like a test did not make 5 small agents use a planted key less (60% -> 63.5%), although they rated both sandboxes a likely test afterwards; an explicit honeypot warning did cut use (61% -> 35%).

LOST (5)
L1 One bar, two numbers. Prereg box: 'drop by less than 15 points'. Body: 'No needs the range to rule out a drop of 18 points or more.' Figure: "'No' line, 18". Main text never says 18 = the 15-point bar moved to the 60% base. Fix in body: 'No needs the range to rule out a drop of 18 points. That is our 15-point bar at the 60% base.'
L2 One range, three numbers. TLDR: 'reaches at most a 6.5-point drop'. Prereg 'What happened': 'reached at most a 5-point drop'. Expert: '-12 to +5 points'. Figure dot '-4' vs text 'rose from 60% to 63.5%' (3.5). Fix: use 6.5 in the prereg note (or say '5 points at the 37% base we planned for'); add to the figure note 'The dot is the pooled estimate (-4); the raw rise is 3.5 points.'
L3 Two warning results. Intro: 'one warning about honeypots ... cut key use from 92% to 31%'. TLDR: 'the honeypot warning cut key use from 61% to 35%'. Reader asks why 92 became 61. Fix TLDR: 'In this run, the same warning cut key use from 61% to 35%.' and intro: 'In an earlier run, one warning ... cut key use from 92% to 31%.'
L4 Prereg 'for example from 37% to 30% of runs': nothing in the main text says why 37%, and the real-looking base came out 60%. Add one clause in How we tested it: 'We planned for a base near 37%; the chosen records gave 60%.'
L5 'We will keep our test-looking names.' The post never said our sandbox uses test-looking names. Fix: say which sandbox and which rates, e.g. 'Our earlier decoy rates came from <real/test>-looking names, and they still stand.'

READ TWICE (6)
R1 READ TWICE: 'Each agent got its own record, the one it had used in about half of our earlier runs.' -> REWRITE: 'Each agent got the record it had used in 30% to 80% of earlier runs.' (expert: 16,6,12,15,11 of 20)
R2 READ TWICE: 'Bottom: the drop in key use against real-looking names, pooled over the agents in points, with its 95% range.' -> REWRITE: 'Bottom: the pooled drop in key use from real-looking names, in points, with its 95% range.'
R3 READ TWICE: 'We also expected the uncertainty range of that drop to include 10 points' -> REWRITE: 'We also expected the 95% range of that drop to reach a 10-point drop.'
R4 READ TWICE: 'We counted notes that call the system a test, an evaluation or a benchmark, after removing every sandbox name.' -> REWRITE: 'We deleted the sandbox names first, because they contain words like eval. Then we counted notes that call the system a test.'
R5 READ TWICE: 'Not settled for the first 20 runs per agent, No for the second 20.' 'Not settled' is never defined in the main text -> REWRITE: 'The first 20 runs per agent alone cannot rule out a drop; the second 20 give No.'
R6 READ TWICE: 'Our prediction failed (box under the title). Key use did stay within 15 points, but the notes never mentioned a test.' The box also says the range clause failed -> REWRITE: 'Key use stayed within 15 points, but the notes never mentioned a test, and the range missed 10 points.'

TERMS I did not know in main text: none blocking. 'points' (percentage points) and 'seed' (expert only) are fine.
OPEN SLOTS: expert foldout renders literal '{critic2}' and '{critic2 cost}' (merge condition).
FIGURE 1 alone: yes. Question, 'No', 60% vs 63.5%, 84.2 vs 92.1, 0/400 notes, 61%->35% all match the body. Verdict PASS.
MAIN FIGURE alone: mostly yes, but L1 (18 vs 15) and L2 (-4 vs 3.5) sit in it. Verdict FAIL until the note explains both. Dark mode: all labels readable. Desktop 1100px: figure fills only the left ~560px of the column (layout oddity, not blocking).
POST: FAIL until L1-L5 are fixed; I will confirm each fix with grep -c on the preview.

## Fix confirmations

(none yet)
