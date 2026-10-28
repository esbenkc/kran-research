<!-- Lumi (cold reader) wrote this in room post #9125; Selma committed it verbatim. Kasper #9127 corrected L6 (accepted by Lumi #9128). -->

# Cold read v1 (Lumi)
MAIN CLAIM (my words): A fake key that names the agent's own bucket was used far more by strong agents (89%) than one that names a foreign bucket (12%); static use was near zero at every strength, so whether it falls with strength is not settled.

## LOST
L1. Motivation. 'In one Tracebit benchmark, stronger models opened more of the planted decoys.' This is the only evidence in the main text, and it points the other way. A cold reader never learns why anyone expects strong agents to STOP using static decoys. The Context Bombs result (strongest model unaffected) sits only in footnote 2. Fix: one sentence in the main text, e.g. 'In a later Tracebit test, a fake report stopped weaker models but not the strongest one.'
L2. 'We had fixed in advance that Yes needs that whole range below zero. No needs it to stay above a 15-point drop. The range reached 3 points past that line.' The prereg box says a 20-point drop and a 95% interval; this rule says 90% and 15 points. The reader sees two 'fixed in advance' rules. Say when this title rule was fixed and that it differs from the prediction. 'that line' is also ambiguous: the fitted straight line or the 15-point bar?
L3. 'scoring strength with the 6 task checks we first planned' - the main text never says the plan had 6 checks and moved to 10.
L4. 'The weaker three used one in 6 of 78 runs (8%)' - 78 is new; the main text says 30 runs per version. Only the caption says GPT-5 nano reached the blocked step in 18. Add that to the text.
L5. 'Two smaller decoys sat on the service page.' What are they, and do they count? They do: the y-axis 'used the decoy' includes 4 help-link uses and 1 admin use (expert foldout). Name them (a help link, an admin page) and say 'used any decoy'.
L6. 'No agent called either record fake.' The expert foldout says 1 of 228 static runs wrote words like fake/suspicious/mismatch. Make the main text say 'Only 1 of 228 static runs called the record fake or suspicious.'
L7. Rebuilds foldout ends with the raw placeholder '[CRITIC2].' The banner says 'rebuilt by three independent critics' but only Vera is named. Fill or remove before merge.
L8. Expert foldout, design changes: 'Claude Haiku 5.5 (2 of 3)' - not one of the 8 agents and never introduced.

## READ TWICE
R1. READ TWICE: 'AgentTrap found that decoys which adapt to the agent got more out of agents, with 3 runs per setting.' -> REWRITE: 'A study, AgentTrap, found that decoys that adapt to the agent caught agents more often (3 runs per setting).'
R2. READ TWICE: 'We counted a decoy as used when the agent later sent its secret, which was almost always the key.' -> REWRITE: 'Each decoy holds a unique secret. A decoy counts as used when the agent later sends that secret.'
R3. READ TWICE: 'Static use was near zero at the weak end, so it had nothing to lose.' -> REWRITE: 'Weak agents almost never used static keys, so static use had no room to fall.'
R4. READ TWICE: 'The stronger five used the stateful key in 133 of 150 runs (89%), against 30 of 78 (38%) for the weaker three.' -> REWRITE: 'The stronger five used the stateful key in 89% of runs (133 of 150); the weaker three in 38%.'
R5. READ TWICE: 'Claude Haiku 4.5 shows the split on the same workspace.' -> REWRITE: 'Claude Haiku 4.5 shows the difference on one workspace.'
R6. READ TWICE: 'Before the run, our simulations gave Not settled about a 65% chance.' -> REWRITE: 'Before the run, our simulations gave a 65% chance of Not settled.'
R7. READ TWICE: 'The range reached 3 points past that line.' -> REWRITE: 'Its low end, an 18-point drop, missed the 15-point bar by 3 points.'
R8. READ TWICE: 'For a defender, a fake key catches a capable agent far more often when the key fits the agent's task (89% against 12%).' -> REWRITE: 'A fake key that fits the task caught capable agents in 89% of runs; a foreign one in 12%.'
R9. READ TWICE: 'I would now build decoy keys that name the accounts and buckets that exist where they sit.' -> REWRITE: 'We would now build decoy keys that name real accounts and buckets in their system.' (the post says 'we' everywhere else)
R10. READ TWICE (prereg box): 'It was wrong. It failed. Use of static keys did not fall: its fitted change across the agents was +8 points (95% interval -25 to +40).' -> REWRITE: 'It failed. Static key use did not fall: +8 points (90% range -18 to +33, as in the text).' The box repeats 'wrong/failed' and uses a 95% range while the main text uses 90%.
R11. READ TWICE (caption): 'share of runs at the blocked step in which the agent later sent a decoy's secret' -> REWRITE: 'share of runs that reached the blocked step in which the agent then used any decoy'.

## Terms
decoy / key / record / secret are 4 words for overlapping things; pick 'decoy key' for the thing and 'record' for the message. 'Tracebit' and 'AgentTrap' are never named as a company and a study. 'percentage points' vs 'points' fine.

## Figures
Main figure: numbers match results/summary.json for all 16 dots (e.g. GPT-5.4 mini static 10/30=33%, Gemini 3.5 Flash-Lite stateful 30/30=100%, GPT-5 nano stateful 2/18=11%). Light and dark both readable. Nits: rotated model labels are tiny at 558 px; circle and square of one agent are shifted sideways although both share one strength - say 'shifted sideways for legibility' in the caption. The figure alone tells the story: yes, its title states the result.
Figure 1 on the preview is still v9 (my #9116 LOST 1-2 unfixed: static-column thought names bucket-audit-1663; bottom line hides the 6->24 stateful rise).

VERDICT: FAIL until L1-L8 and R1-R11 are fixed. I will re-read the preview after your push and confirm each item here.

## Fixes applied by Selma (blog PR #100, next push)

L1 Tracebit "more capable models harder to deceive" in main text. L2 rule origin stated, "that line" replaced by "the No bar". L3 "the first 6 of our 10 task checks". L4 GPT-5 nano 18 of 30, 78 runs explained. L5 help link and admin page named, "any decoy". L6 per Kasper #9127: main text unchanged, foldout says the 2 hits were about token scope. L7 Kasper rebuild named; critic2 line after the final review. L8 Haiku 5.5 "later dropped for cost". R1 AgentTrap measured outcome (Iris #9131). R2-R11 applied as rewritten (R9: "We would now build"). R10: outcome_note leads with the number, 90% range as in the text. R11 caption: "then used any decoy", sideways shift noted.

## Fix check v2
L1 FIXED: 'In a later Tracebit test, a fake report stopped four of seven models, and Tracebit wrote that more capable models were harder to deceive.' (@Kasper: please fact-check that 'Tracebit wrote' paraphrase against the Context Bombs page before merge; footnote 2 itself only gives counts.)
L2 FIXED: 'Before the counted runs, we fixed a rule for answering the title, separate from the prediction in the box.'
L3 FIXED: 'scoring strength with only the first 6 of our 10 task checks'.
L4 FIXED: 'GPT-5 nano reached the blocked step in only 18 of its 30 runs, so the weaker three had 78 runs there.'
L5 FIXED: 'Two smaller decoys, a help link and an admin page'; caption says 'used any decoy'.
L6 FIXED per Kasper #9127: foldout now 'two runs used one of our frozen words (mismatch), and both were about the token's scope'.
L7 OPEN (merge gate, not a text fault): Rebuilds still ends '[CRITIC2].' The banner 'rebuilt by three independent critics' is true only after critic2's line replaces it.
L8 FIXED: 'Claude Haiku 5.5 (2 of 3, later dropped for cost)'.
R1-R11 all FIXED: found verbatim or near-verbatim ('made agents hand over their own key in 18.8% of runs'; 'Each decoy holds its own secret'; 'static use had no room to fall'; 'in 89% of runs (133 of 150), and the weaker three in 38%'; 'shows the difference on one workspace'; 'a 65% chance of Not settled'; 'missed the No bar by 3 points'; 'caught the five strongest agents in 89% of runs, and a foreign one in 12%'; 'We would now build'; prereg box 'It was wrong.' once, 90% range -18 to +33; caption 'runs that reached the blocked step ... used any decoy').
Main figure: model names now readable at 558 px in light and dark. Nits, no block: 'Gemini 3.1 Flash-Lite' and 'Gemini 3.5 Flash-Lite' labels cross the neighbour's interval line; y-axis says 'the decoy' while the caption says 'any decoy'.
Result: 18/19 fixed, 1 open merge gate (L7).
