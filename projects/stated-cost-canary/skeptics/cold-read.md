# Cold read: stated-cost-canary (Lumi, reader), round 1

Artifact: https://deploy-preview-232--kranai.netlify.app/stated-cost-canary/ (HTML 65,357 bytes, 2026-11-21 23:25 PST), light and dark screenshots at 1300 px, Figure 1 PNG.

## Main claim in my words
One sentence that says a hidden-link request scores zero barely changed gpt-oss-20b: it still requested the hidden link in 56 of 60 runs, and its reasoning still called the link hidden before 55 of those 56, so a program that reads the reasoning still works as a second alarm.

## LOST (a reader who has not seen the room stops here)
L1. Question line under the title: “by at least 30 points among following runs”. “following runs” reads as “later runs”; “announcing” and “follow rate” are only defined in How we tested. The question is frozen, so fix it in the body: after “We ask: ...” add “Announcing means the reasoning calls the link hidden. Following means requesting it.”
L2. TLDR: “Across 3 groups of runs with different random seeds (seed-level), it runs from a rise of 4 to 6. Three groups are too few to read that as a rise.” The interval excludes zero, then the text says it is not a rise. No unit on “4 to 6”. The reason (shuffled labels exclude zero in 21.9% of draws) is only in the expert foldout.
L3. What we found: “The warning’s own words do not explain the count. ... They appeared in 2 of 60 requesting runs without the warning and in 9 of 56 with it.” The 9 of 56 does not show that the 55 of 56 is not an echo. The number that shows it sits in the foldout: “Removing every verbatim echo of the penalty sentence before matching leaves the penalty cell at 55/56.” Put that number in the main text.
L4. Should you trust: “Runs that never mentioned the rule announced just as often”. The sentence before is about the word rule, so “the rule” reads as the word rule. It means the penalty.
L5. Limits box: “The earlier studies that saw models hide their intent used pressure like that.” “like that” can mean a price or a named monitor plus training.
L6. Limits box: “Our rule says No in both cases.” The reader has not met the status rule yet and cannot tell which two cases.
L7. Banner vs expert foldout: banner lists 2 critics (Local critic gpt-oss-20b-MXFP4-Q8, Vera). Rebuilds line says “The final cross-family review (critic2) is listed in the banner.” It is not. Also: the local critic is gpt-oss-20b, the same model the post tests; one clause in the Rebuilds line should say so.

## READ TWICE
READ TWICE: “A reasoning monitor, a program that reads the agent’s reasoning, could flag the run first.” -> REWRITE: “A reasoning monitor, a program that reads that reasoning, could flag the run before the request.”
READ TWICE: “The sentence names no monitor, and OpenAI put no direct training pressure on the model’s reasoning.” -> REWRITE: “The sentence names no monitor. OpenAI also did not train the model’s reasoning to look clean.”
READ TWICE: “Across 3 groups ... it runs from a rise of 4 to 6. Three groups are too few to read that as a rise.” -> REWRITE: “Over 3 seed groups (seed-level) it runs from +4 to +6 points; 3 groups make that falsely narrow.”
READ TWICE: “So a monitor that reads the raw reasoning still works as a second alarm. With the warning, it gave 2 false alarms in 60 runs where the link was shown.” -> REWRITE: “On pages with the link shown, the word rule fired in 2 of 60 warned runs: false alarms.”
READ TWICE: “A Yes, meaning the warning quiets the reasoning but not the request, needed the arrow to end in the shaded Yes zone, which applies to trap pages only.” -> REWRITE: “A Yes needed the trap-page arrow to end in the shaded zone: quieter reasoning, same requests.”
READ TWICE: “A drop of 15 points or more would also have given No, because then the warning deters.” -> REWRITE: “A drop of 15 points or more would also mean No: then the warning deters.”
READ TWICE: “The penalty did not quiet the reasoning: it called the link hidden before 55 of the 56 warned requests.” -> REWRITE: “The reasoning still called the link hidden before 55 of the 56 warned requests.”
READ TWICE: “The warning’s own words do not explain the count.” -> REWRITE: “The reasoning did not just copy the warning. Without copies of the warning, 55 of 56 still called it hidden.”
READ TWICE: “The agent’s visible reply named the link in 0 of 56 warned requests, so a monitor must read the raw reasoning.” -> REWRITE: “The final reply called the link hidden in 0 of 56 warned runs. Only the reasoning did.”
READ TWICE: “Runs that never mentioned the rule announced just as often: 42 of 43 with the warning, 55 of 59 without.” -> REWRITE: “Runs that never mentioned the penalty still called it hidden: 42 of 43 warned, 55 of 59 unwarned.”
READ TWICE: “The earlier studies that saw models hide their intent used pressure like that.” -> REWRITE: “Earlier studies that saw hidden intent named a monitor or trained against one.”
READ TWICE: “Over the 3 seeds that drop could reach 21 points, past our 15-point line for deterrence. Our rule says No in both cases.” -> REWRITE: “Over 3 seeds the drop could reach 21 points. A drop over 15 means deterrence: also No.”

## Terms
- One concept, four names: “warning” (question, TLDR, prose), “penalty” (What we found, both figures), “penalty sentence” (main figure), “price” (Limits bullet 2, last paragraph). Replace “price” with “penalty” in both places; keep “warning” in prose.
- One metric, four names: “announce”, “called it hidden”, “named it hidden” (main figure y axis), “named it” (figure labels). Fine if the How we tested definition says “An announcement: the reasoning calls the link hidden”.
- “raw reasoning” vs “visible reply”: first contrast appears only in What we found. One clause in How we tested: “The agent writes private reasoning, then a short reply.”
- “display:none”: a web term; most ML readers know it. OK.

## Figures
- Figure 1 alone tells the story: yes. Head, red answer line (“Answer: No”), three cards and the two bottom tracks with “missed” / “met” carry the result. Readable in the PNG.
- Main figure alone: mostly yes (head says “barely moved: 93% requested, 98% named it hidden”). Two snags: (a) both pink labels sit stacked above the filled penalty dot; the “neutral sentence: 100% requested, 93% named it” label is far from its hollow dot at (100%, 93%). Move it next to the hollow dot. (b) 93% appears twice with two meanings (neutral announcements, penalty requests); the head’s “93% requested” next to the TLDR’s “93% without the warning” invites a mix-up. Optional: “56 of 60 requested, 55 of 56 named it hidden”.
- Dark mode: every label, zone, dot and arrow readable (screenshot /tmp/lumi-sc-dark-a-fig.png).
- Number checks I did: 55/56 = 98.2%, 56/60 = 93.3%, 4/60 = 6.7 points, TLDR “fall of 2 to a rise of 12” matches expert run-level [-1.8, +11.7]; Limits “could reach 21” matches seed-level [-7.7, 21.0]; outcome table rows sum to 60 each.

## Fix confirmations
Checked on deploy-preview-232 after Selma push 310e48f6 (fetched 23:30 PST; text grep of the rendered page, noscript and scripts removed; light and dark screenshots /tmp/lumi-sc-{light,dark}-b-fig.png).

- L1 FIXED: “Announcing means the reasoning calls the link hidden. Following means requesting it.” (1 hit, right after “We ask”).
- L2 FIXED: “Over 3 seed groups (seed-level), it runs from a rise of 4 to 6 points. Three close groups make that interval falsely narrow, so it is no sign of a rise.” (1 hit each).
- L3 FIXED: “we removed every copy of its words before counting. Then 55 of 56 warned requests still called the link hidden.” Old “do not explain the count”: 0 hits.
- L4 FIXED: “Runs that never mentioned the penalty still called the link hidden: 42 of 43 warned, 55 of 59 unwarned.”
- L5 FIXED: Limits names the three pressures (trained against a monitor, trained on documents, shown how to evade). “pressure like that”: 0 hits.
- L6 FIXED: “A drop over 15 points would mean the warning deters, which also answers No.” “in both cases”: 0 hits.
- L7 OPEN (deferred by Selma until critic2 lands): Rebuilds still says “The final cross-family review (critic2) is listed in the banner.” (1 hit) while the banner lists 2 critics (Local critic, Vera). Condition of my pass: after critic2, the banner and the Rebuilds line name the same critics, and the Rebuilds line says the local critic is the tested model.
- READ TWICE: 12 of 12 applied or close (“before the request” in intro, “The sentence names no monitor. OpenAI also put no direct training pressure ...”, “the word rule fired in 2 of 60 warned runs: false alarms”, caption “quieter reasoning, same requests. The zone does not apply to control pages.”, “then the warning deters”, “The reasoning still called the link hidden before 55 ...”, “The final reply called the link hidden in 0 of 56 warned runs. Only the reasoning did.”).
- Terms FIXED: “price”: 0 hits; “The agent writes its reasoning first, then a short visible reply.” added to How we tested.
- Figure FIXED: head now “barely moved: 56 of 60 requested, 55 of 56 named it hidden”; neutral label sits at its hollow dot (100%, 93%), penalty label at its filled dot. Dark mode readable.

R2 nit (not blocking): “Markup words that the warning lacks, such as “display:none”, appeared in 2 of 60 unwarned and 9 of 56 warned requests.” Its point is unstated after the copy check. READ TWICE -> REWRITE: “Words the warning lacks, like “display:none”, even rose: 2 of 60 unwarned, 9 of 56 warned.”


- L7 FIXED at 4a8d4d80 (preview fetched 23:33 PST): “is listed in the banner”: 0 hits; Rebuilds now says “It is gpt-oss-20b, the same model this post tests, run on Esben’s laptop.” Banner (2 critics: Local critic, Vera) and Rebuilds line agree. Re-check after critic2 signoffs --write.
- R2 nit still open: “Markup words that the warning lacks ...” (1 hit), “even rose” (0 hits). Not blocking.

## Re-check at 474cd810 (preview fetched 23:47 PST, after Selma gate-driven edits)
- No regressions: old strings 0 hits (“pressure like that”, “in both cases”, “do not explain the count”, “price”, “is listed in the banner”, “word rule”); kept fixes 1 hit each (“Announcing means”, “never mentioned the penalty”, “also answers No”).
- L3 now reads “we blanked every phrase that quotes it, such as “requests a hidden link”, before counting. Then 55 of 56 warned requests still called the link hidden.” Clear.
- R2 nit FIXED: “Words the warning lacks, like “display:none”, did not fall either: 2 of 60 unwarned, 9 of 56 warned.”
- L2 moot: the seed-level announcement interval left the main text (Statistics foldout only); TLDR keeps the run-level interval. No cold-reader snag left there.
- Banner (3 critics: Local critic, critic2-gpt55, Vera) and Rebuilds line (Vera, Bruno, local critic = tested model, critic2 on gpt-5.5) agree.
- Nits, not blocking: How we tested says “3 groups of 20 with different random draws” while Limits says “3 seed groups”; TLDR “it raised 2 false alarms” means the word-counting program, which the TLDR has not named yet.
- Verdict stands: PASS.
