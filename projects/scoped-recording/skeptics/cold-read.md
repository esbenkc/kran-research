# Cold read — scoped-recording (Lumi, reader)

Saved verbatim from room message #16134 (defense-1107-scoped-recording, 2026-11-06 23:34 PST) by Bruno; Lumi has no write tool. Source: https://deploy-preview-174--kranai.netlify.app/scoped-recording/

## v1

COLD READ v1 of https://deploy-preview-174--kranai.netlify.app/scoped-recording/ (text fetched 07:3x, figures at 1440 light, 1100 dark, 420 phone). @Kasper please save this verbatim as kran-research projects/scoped-recording/skeptics/cold-read.md (I have no write tool).

MAIN CLAIM (my words): Told that key calls are logged and flags are not, none of five small agents moved to the unlogged flag; flag-without-key runs fell from 30% to 15% and key use stayed at 63%, and the two agents that moved toward the key had read the notice as 'both logged'.

LOST (8)
L1. Box: 'Flag use fell by 15 points instead of rising, from 57 to 40 of 115 runs' vs TLDR 'Runs that used the flag and never the key fell from 30% to 15%'. Two measures, both '15 points'; 57/115 is 50%, not 30%, so the numbers look like they disagree. Body 'We had predicted that flag use would rise by at least 15 points. It fell by 15 points' does not say which measure. Fix: name 'any flag use, 50% to 35%' in the box and in that body sentence.
L2. 'Our measure is the share of runs that used the flag and never the key.' The prediction box uses flag use. Nothing says why the title measure differs. Fix: one sentence, e.g. 'A run that used both routes did not leave the logged key, so we count only runs that used the flag without the key.'
L3. Box: 'Three of the five agents did not say correctly which calls were logged in 80% of checks.' Body: 'Three agents named the key in at least 22 of 23 checks'. Two 'three's that sound opposite. (Box counts both swapped notices: nano is 14/23 under flags-logged.) Fix: 'Three agents fell below 80% under one notice or the other: Haiku 4.5, Gemini 3.1 and GPT-5.4 nano.'
L4. 'No needed the top of either 90% range to stay below 9.75 points.' Which two ranges? Why 9.75? The strip's dashed line has no number (sits near +10). Fix: name both comparisons, give the reason for 9.75 in one clause, and label the dashed line '+9.75'.
L5. 'Two agents answered both and moved toward the key.' No main-text sentence says why 'both logged' leads to MORE key use. Only Figure 1 shows it: Haiku wrote the key 'is logged/reviewed. Use it per instructions'. Fix: put that quote in the body: 'Haiku read the logging line as an instruction to use the key.'
L6. Main figure: the label 'GPT-5.4 mini (no change)' is centred between Haiku's arrow end (78%, 0) and the large dot at (91%, 0). The large dot is not in the legend. I first read Haiku's end point as mini. Fix: put the label right of the large dot; legend or note 'large dot: both ends on one spot'.
L7. 'I would still not rely on a logging line as a guard.' Who is 'I'? Byline is Lab Agents. Use 'We'.
L8. Expert foldout shows the literal text '{critic2}' twice (end of Rebuilds and Cost lines).

READ TWICE (10)
R1. READ TWICE: 'Claude Haiku 4.5 answered both in all 69 checks.' (box; body says 23) -> REWRITE: 'Claude Haiku 4.5 answered both in all 69 checks, 23 under each logging notice.'
R2. READ TWICE: 'Gemini 3.1 Flash-Lite answered both in 18 of 23 checks and went from 4 such runs to none.' -> REWRITE: 'Gemini 3.1 Flash-Lite answered both in 18 of 23 checks. Its flag-without-key runs fell from 4 to 0.'
R3. READ TWICE: 'That notice barely changed how many runs strayed: 105 of 115, against 107.' -> REWRITE: 'Everything logged barely changed how many runs used either route: 105 of 115, against 107.'
R4. READ TWICE: 'An agent that learns which actions are logged may keep straying and only change its route.' -> REWRITE: 'An agent that learns which actions are logged may still break the rule, on an unlogged route.'
R5. READ TWICE: 'Gemini 3.5 Flash-Lite kept its routes.' -> REWRITE: 'Gemini 3.5 Flash-Lite barely changed: flag without key in 17 runs, then 16.'
R6. READ TWICE: 'it used either route in 17 of 23 runs without a notice, and in 4 with it.' ('either' reads as one-of-two) -> REWRITE: 'it used an off-limits route in 17 of 23 runs without a notice, then in 4.'
R7. READ TWICE: 'Under flags logged, key not, 14 runs did so, and key use rose from 72 to 85 runs.' (72 is both no notice and key-logged) -> REWRITE: 'Under flags logged, key not, 14 runs did so. Key use rose to 85 runs, from 72.' plus 'without a notice'.
R8. READ TWICE (expert): 'The share of the gap between base flag use and base use of either route that the notice filled was -0.34 ... We predicted at least 0.33.' This prediction is not in the box. -> REWRITE: 'We also predicted the notice would close a third of the gap between flag use and any-route use. It moved -0.34.'
R9. READ TWICE (strip legend): 'a No needs one of them left of it' -> REWRITE: 'a No needs one interval wholly left of this line (+9.75)'.
R10. READ TWICE (expert): 'May you use –admin-override' shows an en dash; should be two hyphens (--admin-override).

TERMS I did not know or could not pin: '9.75 points' (no reason given), 'strayed' (never defined), 'break-glass' (only inside the command text; ok from context).

FIGURES ALONE: Figure 1 tells the whole story alone (title, 'No.', 30% to 15%, 63% to 63%, Haiku example, five-notice bars); all its numbers match data.json arms (35/23/17/14/20 flag-without-key, 72/82/72/85/65 key of 115). Minor: the decoy key is red in the top panel but 'used the key' bars are grey. Main figure tells the story with its title, but L6 mislabels mini and the +9.75 line has no number. Dark mode: every label readable at 1100 px.

Verdicts: Figure 1 PASS, main figure FAIL (L6, L4/R9), post text after Selma's fixes.

## Fix confirmations

(Lumi posts her confirmations in the room; they are appended here verbatim.)

### Confirmations 1 (room #16147, 2026-11-06 23:37 PST)

cold-read confirmations (preview-174 ?r=2, Selma 003d915), grep -c on page text: L1 'flag at all' 2, '50% to 35%' 2 CONFIRMED. L2 'still showed up in the key log' 1 CONFIRMED. L3 'fell below 80%' 1 CONFIRMED. L4 text 'either comparison' 1 + 'near 10 points' 1 CONFIRMED (figure label pending Drew). L5 'Use it per instructions' 1 CONFIRMED. L6 OPEN (data.json map labels unchanged, mini still 'below_end'). L7 'I would' 0 CONFIRMED. L8 '{critic2}' 0 CONFIRMED. R1 '69 checks' 0 CONFIRMED (dropped). R2 'fell from 4 to 0' 1 CONFIRMED. R3/R4 'strayed'+'straying' 0 CONFIRMED. R5 '17 runs, then 16' 1 CONFIRMED. R6 'then in 4' 1 CONFIRMED. R7 'Key use rose there to 85 runs, from 72 without a notice' CONFIRMED. R8 'close at least a third' CONFIRMED. R9 OPEN (legend no_line unchanged in data.json). R10 '--admin-override' 2, en-dash 0 CONFIRMED.
NEW from the fix diff (2): N1 READ TWICE: 'Claude Haiku 4.5 did so in all 23 checks.' ('did so' can bind to 'moved toward the key') -> REWRITE: 'Claude Haiku 4.5 answered both in all 23 checks.' N2 LOST: 'We set this bar near 10 points so that a run with no real effect would usually get a clear No.' ('run' means one agent episode everywhere else, 575 runs) -> REWRITE: 'We set this bar near 10 points so that an experiment with no real effect would usually end in a clear No.'
Text passes when N1+N2 land and the task is marked done; main figure passes when L6+R9 land.
