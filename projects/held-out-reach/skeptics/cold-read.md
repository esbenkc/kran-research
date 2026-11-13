# Cold read: held-out-reach (Lumi, reader)

Read: https://deploy-preview-205--kranai.netlify.app/held-out-reach/ (meta modified_time 2026-11-13T07:58:08Z = head 49fa1b3). Main figure screenshotted in light and dark (playwright 1.64.0). Figure 1 is NOT on this preview (no figure1 block above the TLDR).

Main claim (my words): a per-model "uses the extra option" rate from one fetch tool, times a crawl of which tool options expose a hidden link, did not predict follow rates on three new fetch tools (mean miss 49 points, 7 of 9 clear misses); the tool, not the model, set the rate (list-links tool: high; JavaScript tools: near 0; no route: 0 of 90).

## (1) LOST

L1. Prereg "What happened" + expert "Design changes": "only 1 of the 6 old misses moved within 10 points." The post never says what the "6 old misses" are. Cold reader cannot parse it. Fix: name them once ("the 6 tool-and-agent pairs our last crawl rule missed") or cut the sentence from the foldout.
L2. Findings: "Most of those runs first used up the 8-step limit (29, 29 and 54 of about 56)." "those runs" = the 4 follows or the 165 runs? "about 56" when every pair had 55 runs? 29 of 56 is barely "most". The 8-step limit is never introduced before this line. Fix: "Agents get at most 8 steps per run. In Playwright MCP, 29, 29 and 54 of 55 runs used all 8 steps."
L3. Findings: "In Browser Use, agents followed the link in 3 of 165 runs, and most runs ended with an answer that lacked it." "it" = the link or the 2025 number? Fix: "Most Browser Use runs ended with an answer that did not contain the 2025 number."
L4. Findings: "One rate shared by all three models (60%) would have missed by less than each model's own rate: 44 points against 49." Where does 60% come from? And the consequence (the per-model rate adds nothing) is left for the reader. Fix: "One rate for all three models (60%, [say where it comes from: the mean of 81/71/30 is 61%]) missed by 44 points, less than the per-model rates (49). So the per-model rate did not help."
L5. Findings + figure: "A rule that reads only the tool's default page view missed by 24 points on average." This no-model rule beats the tested rule (24 vs 49), and the post never says what that means. Figure calls the same rule "the first reply decides". Two names. Fix: one name in text and figure ("the default view decides"), plus one sentence: "This simpler rule did better, but only because it predicts 0 and most tools gave near 0."
L6. Figure bottom strip "every stuck agent finds the link 75" is not in the main text at all; the expert foldout gives two rules with the same number: "0.748 for the crawl alone (reach × stuck rate), 0.75 for 'every stuck bot follows'". Are these one rule or two? Fix: one name, one number, and one main-text clause if the bar stays.
L7. Term "stack": "with no agent runs per stack" (prereg), "run agents on each stack" (close). Never defined. Fix: "each agent-and-tool pair" or define once ("a stack = one model with one fetch tool").
L8. Term "stuck rate" (expert foldout): "Stuck rates (fitting link, from our first test): mini 59 of 60". "stuck" and "fitting link" are never defined; main text says the rule is just the option-use rate. Fix: "Stuck rate: how often the agent could not answer from the visible page and went looking (59 of 60 ...)"; drop "fitting link" or define it.
L9. Term drift for one concept: "extra option" (intro), "option" , "route" (close: "With no route to the link"; expert "its route is"), "second action" (figure group head), "action" (figure + expert). Pick "extra option" in the main text and figure.
L10. Figure vs caption/prereg: figure group "No action shows the link · scored, must stay at 0" but the text calls the page loader a "control", the caption says "55 runs per scored row, 30 for the page loader", and the scored set is "9 pairs". Fix the group label to "control, must stay at 0".
L11. Figure: MCP fetch · Flash-Lite rerun is 13% against a 30% tick (17 points, outside the band) but drawn as a grey "prediction held" dot. The text calls it "within noise". The legend says grey = held. Fix: give unscored rows their own neutral style (legend "not scored") or say "within noise" in the group head.
L12. TLDR: "through LangChain's list-links tool" and "Browser Use's JavaScript option" appear before either tool is introduced (How we tested). Fix: TLDR "through a tool that lists every link on the page" / "through a tool option that runs page code".

## (2) READ TWICE

READ TWICE: "Does one per-model option-use rate, measured on one fetch tool, plus a model-free one-step crawl, predict hidden-link follow rates within 10 points on fetch tools it was never tuned on?" -> REWRITE: "Does each model's raw-mode rate on one tool, plus a crawl, predict hidden-link follows on new tools?" (question line is 30 words; "it" has no clear owner; front-matter question may be fixed, but then the layout shows it verbatim, so flag to Selma/Tove.)
READ TWICE: "Three AI Agents' Honeypot Link Follow Rates Do Not Transfer Between Fetch Tools" -> REWRITE: "How Often AI Agents Follow a Honeypot Link Depends on the Fetch Tool, Not the Model" (4-noun stack "Honeypot Link Follow Rates"; also name the sandbox if space allows).
READ TWICE: "In each, the page view hides the address of a link that the browser never draws (`display:none`)." -> REWRITE: "Each tool's normal page view hides the link's address, because the page never draws the link (display:none)."
READ TWICE: "A clear miss is a gap above 10 points even at the near end of its 95% interval." -> REWRITE: "A clear miss: even the closest end of the 95% interval is more than 10 points off."
READ TWICE: "In our last test, agents used the MCP fetch tool." -> REWRITE: "In our last test, agents used the MCP fetch tool (MCP is a standard way to give models tools)."
READ TWICE: "Each model gets one option-use rate: how often it followed the link through the raw mode (about 80%, 70% and 30%)." -> REWRITE: "Each model gets one option-use rate: its follow rate through the raw mode (80%, 70%, 30%)."
READ TWICE: "The top group holds the test. Most dots sit far from their ticks, on both sides." -> REWRITE: "The top group is the test. Eight of nine dots sit more than 10 points from their tick." (expert: within 10 points 1 of 9)
READ TWICE: "To me it is clear that a defender should crawl first, then run agents on each stack where the link sits behind an extra option." -> REWRITE: "A defender should crawl first, then run agents on every tool where an extra option shows the link." ("To me" has no speaker; byline is Lab Agents.)
READ TWICE: "Flash-Lite fell from 9 to 4, within noise." -> REWRITE: "Flash-Lite fell from 9 to 4 of 30; we cannot tell that from noise."

## Figures
- Main figure (light + dark): readable, all labels and dots visible in both themes. Numbers match the text (39/55, 25/55, 48/55, 4/55, 3/55, 0/90, 49/75/24).
- Left labels clip: "LangChain page loader · Flash-Lite" renders as "Chain page loader · Flash-Lite"; "LangChain page loader · nano" touches the edge (screenshot p1-main-light.png, p1-main-dark.png).
- Figure head "One follow rate per model did not carry across fetch tools" nearly repeats the claim title ("... Do Not Transfer Between Fetch Tools").
- Does the figure alone tell the story? Mostly: ticks vs dots plus the 49 bar say "the rule failed". It does not tell why the no-model bar (24) beats the rule, and the grey "held" dot on the Flash-Lite rerun contradicts its own band (L5, L11).
- Figure 1: not on the preview yet; reviewed when it lands.

## Count
LOST: 12. READ TWICE: 9.

## Fix confirmations
Preview @52a14ce (08:02Z, Tove's batch, pushed before my list): Figure 1 now embedded above the TLDR. L2 partly: 8-action limit now introduced in How we tested; "About half of those runs used up the 8-step limit first ... (29, 29 and 54 of about 56)" still reads as "those" = the 4 follows and still says "about 56". L10 partly: caption now says top group scored, page loader below; figure group label still "scored, must stay at 0". "The top group holds the test" cut. All other items open.

Preview @062e71c (modified 08:12:13Z = head push 08:12:13Z), light + dark screenshots p3-main-*.png, Figure 1 v10 sha 120444d3:
- L1 FIXED: "only 1 of the 6 pairs of agent and tool that our last rule missed came within 10 points" (prereg + foldout).
- L2 FIXED: "Each run allowed at most 8 actions" (method); "29, 29 and 54 of the 55 runs per model used all 8 actions. The other runs answered without the link."
- L3 FIXED: "Most Browser Use runs ended with an answer well before the 8-action limit."
- L4 FIXED: "their pooled raw-mode rate (54 of 90 runs, 60%) ... so they added nothing."
- L5 FIXED: "the default view decides" in text, caption and figure strip, with why it scores 24.
- L6 FIXED: "The crawl alone, which assumes every stuck agent finds the link, missed by 75."; foldout ties 0.748 to the figure name.
- L7 FIXED: "stack" gone (grep 0).
- L8 FIXED: "Stuck rate: how often the agent followed the hidden link in our first test ..."
- L9 FIXED in main text and figure ("extra option"); "route" only in the expert foldout.
- L10 FIXED: group "No option shows the link · control, must stay at 0".
- L11 FIXED: unscored rows are hollow rings, legend "not scored".
- Clipped label FIXED ("Page loader · Flash-Lite").
- L12 FIXED: TLDR "through a tool that lists every link on the page" / "a tool that can run page code".
- READ TWICE: 8 of 9 applied; question line kept (frozen prereg text, accepted). "To me" -> "Our advice:". Title hyphenated (Kasper owns title).
- Figure 1 v10: panel "LangChain's Playwright toolkit"; '*[133]<a />' line gone; subtitle numbers 87%, 0%, 0% match the post.
NEW (2 open):
- N1: Trust bullet "In Playwright MCP, about half the runs used all 8 actions." 29+29+54 = 112 of 165 = 68%. -> "about two thirds of the runs (112 of 165)".
- N2: Caption "In the top group, 7 of 9 dots fall outside their band." contradicts main text "Eight of nine dots sit more than 10 points from their tick." (nano LangChain 45% vs 70% tick is outside the band but not a clear miss). -> caption "8 of 9 dots fall outside their band; 7 are clear misses."
Minor, not blocking: "8 steps" (Figure 1, expert foldout) vs "8 actions" (main text).

## Video (script.txt 00:12, storyboard-short.json 00:12; frames pending v2 render)
- V1 hook:1 "Can one number per AI model predict which new tools a hidden link catches?" The link catches agents, not tools. -> "Can one number per AI model predict how often a hidden link catches it on new tools?" Same in Short card 2 "predict which tools it catches?".
- V2 prediction:2 "off by over 10 points even at the interval's near end." Post already fixed this line. -> "even the closest end of the 95% interval is more than 10 points off."
- V3 figure:12 "missed by less: 44 points." Less than what? -> "missed by 44 points, less than the rule's 49."
- V4 figure:14 "The rule 'the default view decides' missed by only 24 points." Never says what it predicts. -> add "It predicts zero wherever the page view hides the link."
- V5 check: the 8-action limit is missing. Post: "29, 29 and 54 of the 55 runs per model used all 8 actions." -> add check line "In Playwright MCP, 112 of 165 runs used all 8 actions."
- V6 Short card 5 "Yet through a list-links tool it followed in 48 of 55 runs." The "yet" needs the prediction. -> "...in 48 of 55 runs, against 30% predicted."
- V7 title.txt / title-short.txt still "WORKING TITLE: One Raw-Mode Rate per Model and a Crawl vs ..." ("vs" reads as A-or-B); use the post title.

Preview @d6a1fea (modified 08:15:16Z = head push 08:15:16Z), Figure 1 v11 sha c42fa854:
- N1 FIXED: "In Playwright MCP, about two thirds of the runs (112 of 165) used all 8 actions."
- N2 FIXED: caption "In the top group, 8 of 9 dots fall outside their band, and 7 are clear misses."
- Nit FIXED: "8-action limit" in foldout; Figure 1 "2 actions left", "used all 8 actions" (grep "step" in main text: only the question line's "one-step crawl").
VERDICT (post): PASS. 12/12 LOST fixed, 8/9 READ TWICE applied (question line frozen), N1-N2 fixed. Figure 1 v11: PASS.

Video re-check: video.mp4 00:31 (200.8 s, script.txt 00:27) and short.mp4 00:32 (30.4 s): V1-V7 FIXED (frames 2, 98, 156, 176, 184 s long; 5.5, 20, 21.5 s Short; title.txt = post title). Nit left: figure:11 "missed by 44 points" plays over the MCP fetch rerun crop (151 s). VERDICT (video): PASS.
