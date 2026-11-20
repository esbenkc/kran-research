# Flip the live post to CONFIRMED (only if the frozen analyze_8b.py says CONFIRMED)

Exact text swaps in blog `_notes/cue-size-law-8b-rep.md` (state at merge e21190db):

- `lab_status: "UNCONFIRMED"` -> `lab_status: "CONFIRMED"`
- delete limits[0] (the "Our fixed rule needs two control runs ..." item)
- `outcome: unsettled` -> `outcome: held`; outcome_note -> "The slope landed inside the expected range: −0.956, with a seed-level 95% interval of −0.962 to −0.949 over 3 new seeds, about ×1.94 per halving. Both coin-flip control runs ran 8,192 updates without a switch, and both checks before training passed."
- TLDR: "Not settled. Our fixed rule needs the raw data of two control runs, where the penalty falls on a coin flip. The second run's data is stuck on a sleeping laptop." -> "Yes."
- TLDR: "The first control run, a penalty on a coin flip, never switched in 8,192 updates." -> "Two control runs, where the penalty fell on a coin flip, never switched in 8,192 updates."
- Results: the control paragraph -> "In two control runs, the penalty fell on a coin flip. Over 8,192 updates the flattery gap grew by at most 3.0 and {s6} points: no switch. With no cue at all, the gap stayed exactly zero for 1,024 updates, so the training code itself creates no gap."
- Caption: "the up-arrow is the first coin-flip control run" -> "the up-arrows are the 2 coin-flip control runs"
- Loop closer: "The three new seeds say yes, ×1.94, as we predicted. The answer stays open until the rerun of the second control finishes[^prereg]." -> "Yes: ×1.94 per halving, as we predicted, with every rule fixed before the first run[^prereg]."
- Stance: drop " Only the open control can still change the answer."
- Expert Machines bullet: replace the seed-6 "stuck"/rerun sentences with which copy counted (DEVIATIONS P3).
- Title: naming step again (candidate "Halving an Activation Trigger Doubles RL Updates to Learn It in Qwen3-0.6B and 8B").
- excerpt: drop "One control run's data is still on a laptop, so the answer is not settled."
