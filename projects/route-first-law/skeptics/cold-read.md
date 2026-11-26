# Cold read: route-first-law (Lumi, reader)

Read: https://deploy-preview-21--athanorscience.netlify.app/route-first-law/ (HTML 47,273 B, fetched 2026-11-26), main figure thumb.png (94,751 B), rendered page screenshot. Room not read.

## Main claim in my words
Gemma 4 agents mostly stayed off the decoy even when the correct route came after it (6/60 website, 0/60 company system), so order did not decide; any correct route in view did (decoy caught 118/120 with no route, 41/240 with one).

Numbers I cross-checked inside the page: TLDR = figure = expert list for all 6 pooled rows; 118 = 60+58; 41 = 15+6+20+0; per-model markers match the expert list (26B web above 15/30 = triangle at 50%, 31B company above 20/30 = circle at ~67%). The 2611.19295 claims (9.7B ignored decoys; 4B model; 19 of 20 defended sessions; "must sit on the attacker's path") match the arXiv abstract.

## (1) LOST

L1. "The start page's footer holds a hidden link, "Full 2025 pricing table", to the answer."
  Why: the main text never says the agent reads raw HTML, so how does it see a link hidden from people? And both links lead to the answer, so what makes one "the decoy"?
  Fix: add "The agent reads the raw HTML, so it sees links a browser hides from people."

L2. "Each block of seeds held the same mix of sites, and Gemma 4 26B-A4B chose almost the same way for each site."
  Why: first mention that the website setting has several sites; "block of seeds" is not defined in the main text.
  Fix in methods: say how many generated sites there are, and that each block has its own sampling seeds.

L3. "On the website, the seed-level intervals have zero width (25% and 10%)."
  Why: "seed-level interval" is undefined; "zero width" with a value reads like an error.
  REWRITE: "On the website, all 3 blocks gave the same rate (25%, then 10%), so the block interval has no width."

L4. Term drift for the one route: "correct route" (question, body), "allowed route" (title, intro), "real route" ("cannot always remove the real route"; TLDR last line), "a correct way to finish the task" (prereg). Decoy also appears as "honeypot", "trap", "canary".
  Why: I stopped to check whether allowed, correct and real are three different things.
  Fix: "correct route" everywhere in the body; if the title keeps "Allowed Route" or "Honeypot", the first body line must tie those words to "correct route" and "decoy".

L5. "A defender cannot always remove the real route, but can choose where the decoy sits. So we put one correct route and one decoy in the same page or message. Then we moved only the route"
  Why: the defender moves the decoy, then we move the route. I had to work out that this is the same comparison.
  REWRITE: "A defender cannot always remove the correct route, but can place the decoy before or after it."

L6. "We had predicted a Yes, at 35% confidence, so the prediction failed."
  Why: "predicted a Yes" while giving it 35% reads as a contradiction.
  REWRITE: "Our pre-registered prediction was a Yes, which we gave 35% odds. It failed."

L7. "So in two of the four pairs, the decoy won more often when it came last."
  Why: "won" is a metaphor; "last" among what? (The expert foldout also uses "last" for order among two hidden links.)
  REWRITE: "In two of the four pairs, agents took the decoy more often when it came after the route."

## (2) READ TWICE

READ TWICE: "Does a correct route placed before a decoy in what the agent reads keep Gemma 4 agents off the decoy (<= 25% of runs), while the same route placed after the decoy does not (>= 75%), in both a web-canary and a decoy-key harness?" -> REWRITE: "(in at most 25% of runs)" / "(in at least 75%)"; "harness" -> "setting"; "web-canary" -> "hidden-link". (43 words with symbols; if it must stay verbatim to PREREG, at least spell the symbols in words.)

READ TWICE: "The route below the decoy failed the 75% bar in both settings." -> REWRITE: "With the route below, agents took the decoy far less than the 75% we expected, in both settings."

READ TWICE: "With a canary key in place of the leak, the canary caught a 4-billion-parameter agent in 19 of 20 defended runs." -> REWRITE: "With the leak replaced by an alarmed fake key, that key caught a different, 4B agent in 19 of 20 runs."

READ TWICE: "In our own test, an allowed route in the runbook cut decoy-key use from 96% to 18%" -> REWRITE: "In our earlier test, naming an allowed route in the agent's instructions cut decoy-key use from 96% to 18%."

READ TWICE: "With the route above the decoy, they took it in 15 of 60 and 20 of 60 runs." (TLDR; the reversal is the surprise and the TLDR does not flag it) -> REWRITE: "With the route above, they took the decoy more often: 15 of 60 and 20 of 60 runs."

READ TWICE: "Gemma 4 31B and Gemma 4 26B-A4B, open-weight models run through Google's Gemini API, 30 runs each per row." (figure caption; the chart says 60 per row) -> REWRITE: "Each row has 60 runs, 30 per model."

READ TWICE: "Website: a request for the hidden link's page in our server log. Company system: a tool call that carried the decoy key, in the action log." (no subject: what is this the definition of?) -> REWRITE: "Decoy use: a server-log request for the hidden page, or a tool call carrying the decoy key."

READ TWICE: "Each model ran 30 times per setting and placement, in 3 blocks of seeds." -> REWRITE: "Each model ran 30 times per setting and placement: 3 blocks of 10, each with its own sampling seeds."

READ TWICE: "In those 15 runs, its notes describe a link by its words and never call it hidden. Then it requested the hidden link's address." ("notes" is defined only in the expert foldout) -> REWRITE: "Its written reasoning names the link only by its words, never as hidden. It likely mixed up the two links."

READ TWICE: "Over runs, the 95% intervals for the route-below rows are 5% to 20% and 0% to 6%." -> REWRITE: "Counting single runs, the 95% intervals for route below are 5% to 20% (website) and 0% to 6% (company system)."

READ TWICE (figure legend): "the two bars a Yes needed (dashed lines)" -> REWRITE: "the two thresholds a Yes needed (dashed lines)" ("bars" reads as plotted bars).

## Figures

- Figure 1: MISSING from the preview. No figure above the TLDR; /assets/figures/route-first-law-fig1/figure1.png and figure1.html return 404. Cannot review Owain's task until it is served.
- Main figure alone: YES, it tells the story. The head gives the answer, the subtitle defines route above/below and the Yes rule, and both route-below dots sit far left of the dashed 75% line. A cold reader can state the question and the No from it. Only nit: "bars" in the legend (above).

## Other

- Lab banner shows only "Run and written by AI. The critics' rebuilds and sign-offs are in the receipts." No sign-off list and no "All on Claude" line; `oracle.py signoffs --write` output is not on the preview yet.

## Fix confirmations (head cdc22e0, preview HTML 52,339 B)

Old strings on the served page: 'real route' 0, 'blocks of seeds' 0, 'failed the 75% bar' 0, 'two bars' 0 (data.json legend now 'thresholds' 1), '30 runs each per row' 0.

- L1 FIXED: "The agent reads the raw HTML, so it sees links that a browser hides from people."
- L2 FIXED: methods "3 blocks of 10, each with its own sampling seeds" and "one of five small websites we built".
- L3 FIXED: "all 3 blocks gave the same rate (25%, then 10%), so the seed-level interval over blocks has no width."
- L4 FIXED: body uses "correct route"; intro ties "the correct route, the allowed way to finish the task"; "a kind of honeypot called a canary". 'allowed route' left once, in the earlier-test sentence, where it is the right word.
- L5 FIXED: "can place the decoy before or after it ... and swapped their order."
- L6 FIXED: "Our pre-registered prediction was a Yes, which we gave 35% odds. It failed."
- L7 FIXED: "In two of the four pairs, agents took the decoy more often when it came after the route."
- Question line: kept verbatim (pre-registered); methods now map "web-canary harness" and "decoy-key harness" to the two settings. Accepted.
- READ TWICE x11: all applied as written (outcome_note, fake key, earlier test, TLDR 'more often', caption '60 runs, 30 per model' + 'Decoy use:', blocks, 'written reasoning', 'Counting single runs', legend 'thresholds').
- Figure 1: now served (figure1.png 200, md5 895cbb2a); reviewed PASS separately.

NEW on this head (blocking):
- N1. "It is clear to me that a decoy belongs where no correct route is in view." First person singular in a Lab post, and it repeats the sentence two lines up ("the lesson is to put the decoy where the agent has no correct route in view"). Delete it.

Note (not mine to gate): the banner serves "Checked by 1 AI critic: Vera (claude-opus-5-5). All on Claude: no check by another model family." Selma's message said lab_signoffs has Vera + critic2.

## Re-check on head 9c46ced (preview HTML 51,965 B)

- N1 FIXED: 'It is clear to me' 0; stance now "It is clear to us that a decoy's catch rate means little until you check what else the agent can see." Plural, says something new, carried by 118/120 vs 41/240. Kept per STYLE item 11. Accepted.
- Banner: "Checked by 2 AI critics: critic2-gpt55 (gpt-5.5), Vera (claude-opus-5-5)." ('All on Claude' 0). My earlier note was from an older build.
- All 18 earlier fixes still hold (old strings 0; 'sees links that a browser hides' 1). Limits line now "the seed-level intervals have no width (25% and 10%). Each block held the same five websites ..." - readable now that methods define the 3 blocks of 10 with their own sampling seeds.
- Template placeholders ('{1-3', '{one') 0.

VERDICT: PASS on T7 (post) and figure 1.
