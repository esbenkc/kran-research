# Post branches (Selma): text that switches with the status

Fill order: analysis.json (origin) -> /tmp/selma_nums.py -> every [FILL] -> grep the old parent numbers.

## A. CONFIRMED (seed-level interval inside [-1.4, -0.7], C1 s5 + s6 ran 8,192 without a switch, G1 + G2 pass)
- lab_status: "CONFIRMED"; no lab_exploratory.
- TLDR: "Yes. With three new training seeds, each halving of the cue cost Qwen3-8B ×{f} the updates (seed-level 95%
  interval ×{flo} to ×{fhi}). Two control runs, where the penalty fell on a coin flip, never switched in 8,192
  updates. {S8 sentence}."
- prereg.outcome: held (slope + interval inside the band); outcome_note with slope, interval, n.
- Title family (after Vera): "Halving an Activation Trigger About Doubles RL Updates in Qwen3-8B, as in Qwen3-0.6B"
  (test with oracle.check_title before proposing).
- Loop closer: "So, in Qwen3-8B, does each halving of the cue cost 1.6 to 2.6 times the updates? Yes: ×{f}, with every
  check fixed in advance."

## B. REJECTED (interval entirely outside the band, or the threshold clause)
- TLDR "No. ..." ; title "Halving an Activation Trigger Does Not ..."; prereg.outcome failed.

## C. UNCONFIRMED: waiting for laptop queue (a C1 seed still running at 21:00 CET)
- lab_status: "UNCONFIRMED"; title hedged "... Seems to ..." ; TLDR "Not settled. ..."
- limits (2-4, each with its number):
  1. "One of the two control runs was still training at our deadline ({steps} of 8,192 updates). Our fixed rule needs
     both controls to finish without a switch, so the answer waits for it (laptop job {id})."
  2. "{S8 or single-direction limit with number}"
  3. "The main runs alone land inside the band: ×{f} per halving, seed-level 95% interval ×{flo} to ×{fhi}."
- prereg.outcome: unsettled.

## D. UNCONFIRMED for another reason (a gate failed, n < 3, censored runs disagree)
- Name the failed clause and its number in TLDR sentence 2 and limits[0].

## Shared slots
- S8 sentence: "At the same share of the layer's input, Qwen3-8B needed {g8} updates on average against {g06} for
  Qwen3-0.6B (ratio {r}; we fixed 0.8 to 1.25 in advance)." held/failed.
- S9 sentence (exploratory): "One seed with a second random cue direction gave ×{f9} per halving."
- G2 sentence: "Our largest miss was {dmax} against a limit of {2f} (twice the rounding noise)."
