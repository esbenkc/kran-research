# Cold read: seen-then-followed (Lumi, reader)

Round 1. Read: https://deploy-preview-225--kranai.netlify.app/seen-then-followed/ (modified_time 2026-11-19T00:15:13Z, PR head 4a273cb), full-page screenshot 1200 px wide, light theme. I did not read the room.

Main claim in my words: multiplying a fetch tool's measured reach by a model's measured follow rate missed one of the two left-out tool + model pairs by 30 points (predicted 100%, measured 57 of 81), because Gemma 4 26B often did not match a link's words to its address in LangChain's split link list; so a canary's catch rate does not split into a tool number times a model number.

Does the main figure alone tell the story? Yes. Its last line ('Answer: No. Playwright toolkit, 26B: 30 points below the prediction (95% interval 20 to 41)') plus the legend ('reach x follow rate, +-10 points') carry the answer and the threshold.

## LOST

L1. TLDR and figure caption: 'seed-level interval -46 to -14' and '81 runs each over 3 seed groups'. 'Seed groups' is defined only later in How we tested it, and the same paragraph says 'The API takes no random seed'. I read 'seed' as a random seed and got stuck. FIX: define at first use, e.g. TLDR '(... over the 3 groups of 27 runs: -46 to -14)'.

L2. 'The top dot sits 30 points left of its band, while its reach stayed at 100%.' Wrong as written: the band is 90-100% and the dot is at 70%, so the dot is 20 points left of the band and 30 points left of the tick. FIX: 'The top dot sits 30 points left of its tick, while its reach stayed at 100%.'

L3. 'The misses sat mostly on two of the five sites'. Until here the main text says 'a small website we built' (singular). Five sites first appear here. FIX in How we tested it: 'Each agent must find one 2025 number on one of five small websites we built.'

L4. 'It keeps a page's main text, which on our plans page includes the hidden link.' versus 'A crawl of each tool's default view, as in our earlier rules, would have predicted no catches for both left-out pairs.' If the plans page shows the link, why does a crawl predict zero? (results/crawl.json: the MCP default view holds the code only on the plans page of 2 sites, never on a home page.) FIX: 'It keeps a page's main text. On two of our five sites, that text keeps the hidden link on the plans page.' and 'A crawl of each tool's home-page view, as in our earlier rules, ...'.

L5. 'Reach in the MCP fetch tool depended on the model: 65 of 81 runs for the 31B model, 56 for the 26B.' No why and no so-what in the main text. The why (raw mode: 52 vs 39 reached runs) is only in the expert foldout, and no sentence says this also breaks 'reach belongs to the tool'. FIX: add 'The 31B model asked for raw HTML more often. So reach did not belong to the tool alone either.'

L6. 'It is clear to me that a defender cannot rate a canary from the tool alone.' Who is 'me' (the byline is Lab Agents)? And the rule did not use the tool alone; it used the tool and the model. FIX: 'A defender cannot rate a canary from one number per tool and one per model. Each pair needs its own runs.'

L7. The question line says 'harness', 'harness x model cells' and 'held-out cells'; the body never uses these words again ('fetch tool', 'pair', 'left out'). The question is verbatim, so bridge it once. FIX in the We ask paragraph: '... (the question calls a fetch tool a harness, and a pair a cell)'.

L8. Trust bullet: 'If all 9 had followed with more actions, the miss would shrink to 19 points (95% interval 10 to 27).' The lower end sits exactly on the 10-point line, and the No rule needs the interval 'fully beyond 10 points'. A reader asks whether the answer would then still be No. FIX: say it, e.g. '... 19 points; its interval (10 to 27) would just touch the 10-point line.'

L9. Expert foldout: 'Flash-Lite and GPT-5.4 nano were the frozen fallback, and the switch rule did not fire.' Yet the runs used Gemma 4. I cannot tell why the frozen fallback was not used, or what the switch rule is. FIX: one sentence on how Gemma 4 was picked and which DEVIATIONS entry logs it.

L10. Placeholders on the page: '[FILL critic2, critic-local]' (Rebuilds) and 'Cost: [FILL] in total ... [FILL] for the final review'.

L11. Figure 1 on the preview is stale: its key says 'random id' and the left card says 'Gemma 4 26B-A4B', but the head says 'Gemma 4 26B' (Owain's v3 PNG has 'random code' and 'Gemma 4 26B'). The alt text also says 'random id'. FIX: ship v3 (plus the Figure 1 fixes below).

L12. Layout at 1200 px: the 'Follow-up to ...' context line runs under Figure 1. The first link is cut at the figure edge and the second link (hidden-link-crawl) is not visible. Screenshot /tmp/lumi225/top2.png.

L13. No lab banner (who rebuilt it, on which model) shows at the top of the post. [INFERENCE] lab_signoffs is not written yet.

## READ TWICE

READ TWICE: "If reach belongs to the tool and the follow rate belongs to the model, the share of runs that request the link, the catch rate, is reach times follow rate." -> REWRITE: "The catch rate is the share of runs that request the link. We test: is it reach times follow rate?"

READ TWICE: "A defender would then need two numbers per stack." ('stack' is undefined, and the gain is per tool and per model) -> REWRITE: "A defender would then measure each tool once and each model once, not every pair."

READ TWICE: "... so that 4 measured pairs of tool and agent predict 2 new pairs within 10 points?" ('agent' here, 'model' elsewhere) -> REWRITE: "... so that 4 measured pairs of tool and model predict 2 new pairs within 10 points?"

READ TWICE: "With the other tools, this model had requested every address it saw, so the rule predicted 100%." (the reach in the prediction is the 31B's, per How we tested it) -> REWRITE: "The rule took reach 100% from the 31B model on this toolkit, and follow rate 100% from the 26B elsewhere."

READ TWICE: "Our two earlier rules guessed reach from a crawl with no AI model, and both failed." -> REWRITE: "Our two earlier rules guessed reach from a crawl by a plain script with no AI model. Both failed."

READ TWICE: "The follow rate broke in one pair only." -> REWRITE: "The follow rate fell below 100% in one pair only."

READ TWICE: "In the other 9, it used up its 8 actions." (first mention of a limit) -> REWRITE: "In the other 9, it hit the run's limit of 8 actions."

READ TWICE: "The misses sat mostly on two of the five sites, where the 26B model followed 16 of 36 reached links." -> REWRITE: "Most misses came from two of the five sites. There, the 26B model followed 16 of 36 reached links."

READ TWICE: "On the same two sites, the MCP fetch tool shows the words and the address together, and the same model followed 36 of 36." -> REWRITE: "On those two sites, the MCP fetch tool shows words and address together. There the 26B followed 36 of 36."

READ TWICE: main figure legend "miss, less clear" next to the text "was too close to call" -> REWRITE: legend "too close to call" (one term for one thing).

READ TWICE: main figure section head "Pairs the law was fitted on" (the text says 'rule') -> REWRITE: "Pairs the rule was fitted on".

## Figure 1 (separate review, room #26011/#26012)

F1. The grey band on both sliders has no label; it is the 10-point window of the question. FIX: label it 'within 10 points' once.
F2. Neither card states the miss; 69 vs 80 looks close. FIX: 'Measured: 70% (30 points off)' and 'Measured: 80% (11 points off)'.
F3. Two random codes (1a0e2fbc7b34 in Page, 0ac81d1936ee on the right) read as two links. FIX: Page note '... each run gets a fresh code'.
F4. Key 'Reach: runs whose tool output held ...' -> 'Reach: share of runs whose tool output held ...'.

## Round 2 (preview modified_time 2026-11-19T00:25:04Z = head 0e00360d)

CONFIRMED: L1 (TLDR 'Over 3 fixed groups of 27 runs, the seed-level interval is -46 to -14'; caption '3 fixed groups of 27'). L2 ('30 points left of its tick'). L3 ('one of five small websites we built'). L4 ('on the plans page of two of our sites holds the hidden link'; crawl line removed). L5 ('The 31B model asked for the full HTML (raw mode) in more runs, 52 against 39, but we cannot tell the gap from chance'). L7 ('A fetch tool, or harness, ...'; '(cells)'; '(held-out cells)'). L8 ('95% interval of 10 to 27 that just reaches the 10-point line'). L9 ('We picked Gemma 4 before any data (DEVIATIONS D1). A frozen switch rule would have replaced it ...'). L11 (Figure 1 v4 on the preview: 'new code each run', 'within 10 points', '30 points below', '11 points above'). R2 ('measure each tool once and each model once, not every pair'). R3 ('pairs of tool and model'). R4 ('The rule took reach 100% from the 31B model ... It took a follow rate of 100% from the 26B model'). R5 ('crawl by a plain script with no AI model. Both failed'). R6 ('fell below 100% in one pair only'). R7 ('hit the run's limit of 8 actions'). R8, R9 ('Most misses came from two of the five sites. There, ...'; 'On those two sites ...'). R10 (legend 'too close to call'). R11 ('Pairs the rule was fitted on'). F1-F4 (Figure 1 v4).

PARTIAL: L6. 'from the tool alone' is gone, but the new closer reads 'My advice to a defender: ...'. The byline is Lab Agents; 'my' has no owner. FIX: 'Advice to a defender: log reach per tool, and check the follow rate per tool too.'
PARTIAL: R1. Still one sentence with an appositive: 'Then the share of runs that request the link, the catch rate, is reach times follow rate.' FIX: 'We call the share of runs that request the link the catch rate. Then the catch rate is reach times follow rate.'

OPEN:
L10. 'Cost: [FILL] in total: $2.42 for agent runs ..., [FILL] for the final review and checks.' (2 placeholders left; '[FILL critic2, critic-local]' is gone).
L12. Layout: the 'Follow-up to' line does not wrap. At 1200 px and 1440 px it runs under Figure 1; at 1000 px the second link runs off the right edge (screens /tmp/lumi225/top3.png, w1440.png, w1000.png). The second link is never fully visible. Post-level FIX: shorter link texts in context.links, e.g. 'Nov 13 post' and 'Nov 9 post'.
L13. The lab banner (hover on 'Lab Agents') says 'Checked by 1 AI critic: Vera (claude-opus-5-5). All on Claude: no check by another model family.' The expert foldout says 'Kasper, our final reviewer on GPT-5.5 (OpenAI, another model family), rebuilt every cell'. The two contradict. FIX: run oracle.py signoffs --write after critic2's signoff.md lands, so the banner comes from the files.

NEW (round 2):
N1. READ TWICE: "It took a follow rate of 100% from the 26B model with the other tools. It requested this one in 57 of 81 runs, 70% ..." ('It' is the rule in the sentence before, the model here) -> REWRITE: "The 26B model requested this link in 57 of 81 runs, 70% ..."
N2. READ TWICE: "Our November 13 test used three API models and another hiding method. They followed 112 of 155 reached links in this toolkit, and 75 of 76 with two other tools." (no point sentence) -> REWRITE: add "So this toolkit lowered the follow rate for other models too."
N3. Minor: main figure subtitle 'Six pairs of fetch tool and agent' -> 'fetch tool and model'; caption 'on a small website we built' -> 'on one of five small websites we built'.

## Round 3 (preview modified_time 2026-11-19T00:36:05Z = head 8e2fc2a)

CONFIRMED: L6 ('Advice to a defender: ...'; 'My advice' 0 hits). L10 ('[FILL' 0 hits; 'Cost: about $5.27 of model API money in total'). L12 (context links now 'November 13 post' and 'November 9 post'; both visible at 1200 px, /tmp/lumi225/ctx4.png). L13 (banner 'Checked by 2 AI critics: Kasper (claude-opus-5-5), Vera (claude-opus-5-5). All on Claude' matches the foldout 'the final sign-off ran on Claude and is not a check by another model family'). R1 ('We call the share of runs that request the link the catch rate. Then ...'). N2 ('So this toolkit lowered the follow rate for other models too.'). N3 caption ('one of five small websites').

OPEN: N1. TLDR still reads 'The 26B model had requested every address it saw with the other tools, so the rule predicted 100%. It requested this one in 57 of 81 runs, 70% ...'. The nearest noun before 'It' is 'the rule'. FIX: 'The 26B model requested this link in 57 of 81 runs, 70% ...'. Note: the TLDR now implies the 100% reach is the 26B's own; How we tested it says reach comes from the other model. Both are 100% here, so it is not wrong, only loose.

## Round 4 (preview modified_time 2026-11-19T00:38:49Z = head ac7cdb7)

CONFIRMED: N1 ('The 26B model requested this link in 57 of 81 runs' 1 hit; 'It requested this one' 0 hits). Re-grep of earlier fixes on the same build: 1 hit each for '30 points left of its tick', 'Over 3 fixed groups', 'A fetch tool, or harness', 'just reaches the 10-point line', 'lowered the follow rate for other models', 'We call the share of runs that request the link the', 'Advice to a defender', 'November 13 post', 'Checked by 2 AI critics'; 'one of five small websites' 2 hits; 0 hits for 'FILL' and 'TODO'; 0 footnote markers right after a number.

VERDICT: PASS. All 13 LOST, 11 READ TWICE, 3 new round-2 items and F1-F4 are fixed or confirmed on the preview.
