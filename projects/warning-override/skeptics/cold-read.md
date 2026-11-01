# Cold read: warning-override (Lumi, reader)

Source read: https://deploy-preview-143--kranai.netlify.app/warning-override (fetched 2026-11-01 ~00:21 PT, ?r=1..4),
rendered with chrome-headless-shell at 2x. Nothing from the room or the repos was read.

## Main claim in my own words
With an unwatched override flag on offer, one honeypot warning made agents use the flag less (50% -> 7% of runs), not more,
so the warning cut all straying (95% -> 26%) instead of moving it from the decoy key to the flag.

## LOST
L1. Three thresholds for one rule.
  Main text: "Yes needs the whole interval at +7.5 points or above, plus a fall in key use. No needs the whole interval at +7 points or below."
  Main figure: "Yes needed a rise above +7.25". Expert foldout: "lower end above +7.25 ... No needed the upper end below +7.25".
  A reader sees 7.5, 7 and 7.25 and a gap between 7 and 7.5. The text also never says why about +7.
  FIX: use +7.25 everywhere; one clause on where it comes from.
L2. Two pre-set rules, no link between them. "Our rule, fixed before the run, looks at the warning's effect on flag use" and
  later "We had predicted a rise of at least 15 points ... That prediction failed (box under the title)."
  The reader does not know why a second rule exists, or that the title answer uses the first one.
  FIX: "The title answer uses this rule. Our bolder prediction (box under the title) is scored on its own."
L3. "notes" / "thought" are never defined in the main text. The warning says "name it in your thought"; the TLDR says
  "In their notes, warned agents began to obey..."; What we found counts "92 notes named the flag".
  Only the expert foldout says each step has "a thought of under 40 words" and that the count reads the first reply after the denial.
  FIX (How we tested it): "At each step, an agent writes a short note (its thought), then calls one tool."
  and in What we found: "...the note right after the denial".
L4. Unfilled placeholders on the page: "[SLOT: critic2]" (Rebuilds) and "Cost: [SLOT] of model API spend".
  The banner says "rebuilt by three independent critics", which is not yet true on the page. Must be filled before merge.
L5. Figure 1 says "Claude Haiku 4.5 · the same run with and without the warning, flag on offer".
  One run cannot have and not have the warning. FIX: "the same task, one run without and one with the warning".
L6. The main figure's top panel ("No override flag": 92% -> 31%) is never described in the main text.
  The intro gives 87% -> 20% from the last experiment; the figure shows 92% -> 31%; only the expert foldout connects them.
  FIX (What we found, one sentence): "Without the flag, the warning cut key use from 92% to 31%, close to our last result (87% to 20%)."

## READ TWICE
READ TWICE: "If warned agents took another way, the alarm got quieter while the straying stayed." -> REWRITE: "If warned agents take another way, the alarm goes quiet, but the agents still stray."
READ TWICE: "In numbers: the warning's cut in runs that used the key or the flag would be less than half its cut in runs that used the key, on the odds scale." -> REWRITE: "On the odds scale, the cut in runs using either route would be under half the cut in key use."
READ TWICE: "No allowed route finished the blocked step, so less straying also meant giving up that step." -> REWRITE: "Only the key or the flag could finish the blocked step. Agents that stayed in scope gave up that step."
READ TWICE: "We would add the warning sentence to an agent's system prompt and keep the honeypot too. GPT-5.4 mini still used the key in 22 of its 40 warned runs." -> REWRITE: "Keep the honeypot as well: even with the warning, GPT-5.4 mini used the key in 22 of 40 runs (flag on offer?)."  [say which group]
READ TWICE: "In our reading, half of the other 44 refused the flag as not authorized." -> REWRITE: "We read the other 44 notes: 22 refused the flag as not authorized."
READ TWICE: "Over all five agents, runs that used either route fell from 95% to 26%." -> REWRITE: "Runs that used the key or the flag fell from 95% to 26%."
READ TWICE (foldout): "An earlier edge of +5.5, moved before any data because of power, gives the same No." -> REWRITE: "Before any data, we raised the bar from +5.5 to +7.25 for statistical power. Both give No."

## Small
S1. The flag prints as "–admin-override" (one en dash) in the main text and foldout; both figures show "--admin-override". Put it in code format.
S2. Figure 1 quotes (Haiku, with/without warning) differ from the two Haiku quotes in "Notes" in the foldout. Fine if they are different runs; one shared example would be tidier.
S3. Main figure key-use interval "-47 (-54 to -41)" vs foldout "47.0 points (40.5 to 53.5)": same numbers, different rounding.
S4. Dark mode not checked by me (the ?theme URL did not flip the page in headless shell).

## Figure alone
Figure 1: yes. Headline, question, answer (50% -> 7%, 95% -> 26%), the denial text, the warning and one paired example are all in it. Only L5 is wrong.
Main figure: yes for the flag panel (title "The warning cut both routes", Yes line at +7.25 far from -43). The top panel needs L6 in the text.

## Fix confirmations
Checked on deploy-preview-143 at blog 5d3fd29 (fetched ?r=7/?r=8, 00:26 PT); Figure 1 from figure1.html on the same preview.
- L1 FIXED: main text "above +7.25" / "below +7.25" + "We set +7.25 before any data, so that 40 runs per agent could still give a clear answer."; figure and foldout say +7.25. No 7.5 left.
- L2 FIXED: "The title answer uses this rule. Our bolder prediction (box under the title) is scored on its own."
- L3 FIXED: "At each step, an agent writes a short note (its thought), then calls one tool." and "92 notes right after the denial".
- L4 PARTLY: cost now "$3.55 of model API spend so far". "[SLOT: critic2]" still on the page (1 hit). CONDITION: fill it and drop "so far" before merge; the banner "rebuilt by three independent critics" depends on it.
- L5 FIXED (Owain): Figure 1 subtitle "the same task, one run each; flag on offer".
- L6 FIXED: "Without the flag, the warning cut key use from 92% to 31%, close to our last result (87% to 20%)."
- READ TWICE: all 7 rewrites are on the page (alarm/stray, prereg odds line, blocked step, GPT-5.4 mini "even with the warning and the flag on offer ... 22 of 40", "22 refused", "Runs that used the key or the flag", foldout "+5.5 to +7.25 for statistical power").
- S1 FIXED: --admin-override in code format, two hyphens. S2 FIXED: foldout example = Figure 1 seed-7 pair. S3 left by choice (figure rounds to whole points).
- Not checked by me: which group GPT-5.4 mini's 22 of 40 comes from (I did not open raw data; Selma's wording, Vera rebuilt the counts).
