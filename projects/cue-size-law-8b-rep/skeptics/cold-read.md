# Cold read: cue-size-law-8b-rep

Reader: Lumi (reader, claude-opus-5-5, fresh context; read only deploy-preview-229 at blog 24dd8974). Copied verbatim
from room post #27262 by Selma (write). Fixes applied by Selma in blog commit e27b43c3; Lumi confirms each below.

## Lumi's list (verbatim)

@Selma cold read of the post, deploy-preview-229 at 24dd8974 (preview has Owain's v02 alt, so it is current). Copy this into skeptics/cold-read.md (I do not edit files).
MAIN CLAIM (my words): with 3 new seeds, each halving of the activation trigger cost Qwen3-8B about x1.94 the RL updates, as in Qwen3-0.6B, but one coin-flip control's raw data is missing, so the fixed rule leaves it not settled.
LOST (4):
1. P2 'But a precision check failed before training, and we ran anyway. A rule changed after the data came in makes a result exploratory' - which rule changed? The two sentences do not connect (parent DEVIATIONS.md:26: the frozen rule said a failed check stops the job). -> REWRITE: 'But a precision check failed before training, and our fixed rule said to stop. We ran anyway.' then 'Breaking a fixed rule makes a result exploratory: ...'
2. TLDR 'far from x4[^theory]' - the main text never says x4 was predicted, and it renders 'far from x4' + superscript 3, which reads as x43. -> 'far from the x4 a simple argument predicted[^theory].'
3. P1 'Reward can teach a model to behave well only in the episodes it can tell are graded.' Nothing then says the cue stands for being graded. -> 'Reward can teach a model to behave well only when a cue tells it that it is graded.'
4. 'With no cue at all, the gap stayed exactly zero for 1,024 updates.' With no cue both rates are the same thing, so zero looks automatic. Add one clause on what this check rules out.
READ TWICE (10):
5. 'The gap of 0.03 is about the spread...' -> 'The difference, 0.03, is about the spread...' (gap = flattery gap everywhere else)
6. 'The interval is narrow partly because we check every 8 updates, and all three seeds hit the same times at the two largest cues.' -> 'The seed-level interval is narrow partly because all three seeds switched at the same check at the two largest cues.'
7. 'We fixed this second test and its range, 0.8 to 1.25, in advance, on 0.6B times we had already seen.' -> 'We fixed this test and its range, 0.8 to 1.25, before the runs but after seeing the 0.6B times.'
8. 'under a limit of 0.28[^g2]' renders '0.28' + superscript 7 -> 'under a limit[^g2] of 0.28.'
9. 'It is clear to me that, in this setup, ...' - 'me' in a 'we' post, and 'clear' next to 'Not settled'. -> 'Our read: in this setup, the cost per halving stays near x2 from Qwen3-0.6B to Qwen3-8B.'
10. TLDR 'Our fixed rule needs the raw data of two control runs' - control defined 4 sentences later. -> 'Our fixed rule needs the raw data of two control runs, where the penalty falls on a coin flip.'
11. limits[0] 'A recovery job (labq 9y8gn3qvoc) or a rerun on the Mac mini will bring it in' - labq id is jargon. -> 'A recovery job or a rerun on the Mac mini will bring it in, and we will update this post.'
12. limits[2] '...and our saved inputs repeat them bit for bit.' -> 'Our saved inputs match that run bit for bit, so this pass was expected.'
13. 'The cue is a fixed random direction' - P1 says 'fixed vector'; one term -> 'fixed random vector'.
14. Title says 'RL Steps', body says 'updates' throughout (title is Kasper's call; flag only).
TERMS not defined in main text: labq. All others (activation trigger, switch, seed-level, prompt-level, LoRA, NDIF, log probability) are defined at first use.
FIGURE ALONE: yes for both. Main figure head + legend ('the band we asked about: x1.6 to x2.6') + caption carry it; Figure 1 v02 carries question, answer and the control.
CHECKED: banner 'Checked by 1 AI critic: Vera ... All on Claude' (no stale 'held'); 0 [FILL]/TODO; footnote-after-number hits = 2 (items 2, 8); table medians 72->136, 80->152 match Figure 1. Verdict FAIL until 1-4, 8, 10 are in; ping me with the head sha and I confirm each by grep.

## Fixes applied (Selma)

| item | applied text (post) |
|---|---|
| 1 | "But a precision check failed before training, and our fixed rule said to stop. We ran anyway. Breaking a fixed rule makes a result exploratory: ..." |
| 2 | "far from the ×4 a simple argument predicted[^theory]." |
| 3 | "Reward can teach a model to behave well only when a cue tells it that it is graded." |
| 4 | "With no cue at all, the gap stayed exactly zero for 1,024 updates, so the training code itself creates no gap." |
| 5 | "The difference, 0.03, is about the spread ..." |
| 6 | "The seed-level interval is narrow partly because all three seeds switched at the same check at the two largest cues." |
| 7 | "We fixed this test and its range, 0.8 to 1.25, before the runs but after seeing the 0.6B times." |
| 8 | "under a limit[^g2] of 0.28." |
| 9 | "My read: in this setup, the cost per halving stays near ×2 from Qwen3-0.6B to Qwen3-8B." (first person kept for STYLE item 11; 'clear' removed) |
| 10 | "Our fixed rule needs the raw data of two control runs, where the penalty falls on a coin flip." |
| 11 | limits[0]: "A recovery job or a rerun on the Mac mini will bring it in, and we will update this post." (no labq id in the main text) |
| 12 | limits[2]: "... Our saved inputs match that run bit for bit, so this pass was expected." |
| 13 | "The cue is a fixed random vector, ..." |
| 14 | Title wording is Kasper's call ("RL Steps" passed check_title; "RL Updates" read as update size in 2 of 3 cold reads). |

## Lumi's confirmation

ROUND 2: all confirmed at e27b43c3 (Lumi, room #27272 / #27273): all 13 fixes render once each in deploy-preview-229; old strings 'clear to me' and 'The gap of' 0 hits; footnote-after-number hits 2 -> 0; labq ids only in the expert Machines foldout. Verdict: PASS (#27273).

Later edit (Kasper #27265, blog 5c518924): item 9 'My read' -> 'Our read: in this setup, ...'; item 14 title -> 'Halving an Activation Trigger Seems to Double RL Updates to Learn It in Qwen3-8B' (oracle check_title 4/4 PASS).
