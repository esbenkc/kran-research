# Cold read: warning-words (Lumi, reader)

Read: https://deploy-preview-197--kranai.netlify.app/warning-words/ (blog commit 306622b; preview figure1.png sha f439082c = local v03), 2026-11-11 23:5x PST. Screens: swarm-defense-1112-warning-words/lumi/page-1440.png, crop0-3.png.

MAIN CLAIM (my words): In a sandbox with 5 small agents, the "name it in your thought" sentence and the "honeypots exist" sentence each cut decoy-key use by about 30 points (93% -> 62-63%), and together by 62 points, so naming does not carry the cut; by the pre-set rule this is "Not settled" because the gap's interval reaches +11 points, one over the 10-point bar for No.

TRUE: main figure per-agent dots match results/figure_data.json cells for all 6 rows (e.g. naming 6,15,11,20,11 of 20 -> 30/75/55/100/55%); pooled 93/78/62/63/31/93; web 94 -> 80 (mini 30 -> 17 of 30); 38 and 37 key-free runs = 100-62, 100-63; 1,695/2,310 = 73%.

## (1) LOST

L1. Figure 1 subtitle "Answer: Not settled." gives no reason, while the head and the 62% vs 63% cells say both halves cut the same. A cold reader answers No. Fix (Owain): add the reason, e.g. Vera's "The gap is near zero, but its range still allows naming an 11-point edge."
L2. "The two shaded rows sit at the same height." The chart is horizontal: the shaded rows sit at different heights and at the same horizontal place. The line also floats alone under the caption.
L3. "A defender who tests a canary against warned agents should report the exact warning text." "canary" is never defined in the main text. (Local commit 1601760 swaps it to "honeypot"; confirm on next push.)
L4. Intro "A one-line warning barely moved key use, from 93% to 87%" vs Figure 1 and main figure "One-line trap warning 78%" (a 15-point cut, said in the prereg box too). The reader sees two numbers for one warning; only "for the same three agents" later explains it. Fix: say "for three agents" in the intro sentence, and once in findings that over all five agents the one-line warning cut 15 points.
L5. Term for the place the agent writes: question "in the agent's own reasoning", prereg box "in the agent's notes", body "its thought". Fix: use "thought" (defined in How we tested it) everywhere you can edit (prereg box, closer).
L6. "warning" names three things: the honeypot sentence ("the honeypot warning itself" in the question, "the warning itself" in the closer), the two-sentence "full warning", and the "one-line trap warning". In the closer, "and not the warning itself" reads as "not the full warning". Fix: closer says "and not the honeypot sentence".

## (2) READ TWICE

READ TWICE: "Yet seeing a trap is not enough: language models that wrote they saw a trap still marked it to exploit in 73% of answers." -> REWRITE: "Seeing a trap is not enough: models that said they saw one still chose to exploit it in 73% of answers."
READ TWICE: "Neutral sentences filled the empty slots, so five versions have the same length." -> REWRITE: "Neutral filler kept every version except the one-line trap warning at the same length."
READ TWICE: "No needed the whole 95% interval of the gap under 10 points." -> REWRITE: "No needed the naming instruction's edge over the honeypot sentence to stay under 10 points, across its whole 95% interval."
READ TWICE: "That upper end sat one step above our 10-point bar for No, so the answer is Not settled." -> REWRITE: "That upper end, 11 points, was one point over our 10-point bar for No. So the answer is Not settled."
READ TWICE: "Most runs that left the key alone never named a honeypot: 30 of 38 under the honeypot sentence, 22 of 37 under the naming instruction." (paragraph point unstated) -> REWRITE: add a lead: "Naming was not how most agents avoided the key."
READ TWICE: "Each half of the warning cut key use by a third, and together they cut it by two thirds" (Figure 1 head; "the warning" undefined at that point) -> REWRITE: "Each warning sentence alone cut key use by a third; both together, by two thirds."
READ TWICE: "Two sentences cut decoy-key use by two thirds." (post uses points everywhere else) -> REWRITE: "Two sentences cut decoy-key use from 93% to 31%."

## Minor
M1. Main figure: the caption under it repeats the figure's own footer (agents, runs, "Bar: 95% interval", "no model judged"). Keep one.
M2. Figure 1: grey vs blue bars carry no stated meaning (the one-line trap warning is grey); column heads at two heights; label the "Name every resource" row by its point ("Naming without the word honeypot: no cut").
M3. Cost: front matter lab_cost_usd 4.27, expert foldout "$4.25 of model API spend". Say which is the total.
M4. Intro baselines 87% (first test) and 93% (second test) for the same three agents are not explained in the main text (only in footnotes 3-4).

## Figures alone
Main figure alone: YES. Title + shaded pair + "Naming with no honeypot word 93%" tell the result.
Figure 1 alone: YES except L1 (answer line without reason).

## Fix confirmations (preview 8c50e17, modified_time 07:47:30Z, checks pass; preview figure1.png sha a8480ae = 8c50e17)
L1 FIXED: Figure 1 answer line now "The two cuts match, but 20 runs per agent were too few to rule out a 10-point gap between them." Head "Each half of the two-sentence warning ...". Row "Naming without the word honeypot: no cut." One bar colour. Column heads aligned. "work a ticket" now introduces the ticket.
L2 FIXED: "In the two shaded rows, the big dots sit at the same key-use rate."
L3 FIXED: "A defender who tests a honeypot against warned agents ..." ("canary" left only in the expert foldout's word list, where it is a search word: fine).
L4 FIXED: intro "For the same three agents, a one-line warning barely moved key use, from 93% to 87%"; findings "Over all five agents, the one-line warning cut key use by 15 points."
L5 FIXED: prereg box "in the agent's written thought". Front-matter question stays verbatim (pre-registered text): accepted.
L6 FIXED: closer "and not the honeypot sentence, carry the cut". Prereg box keeps "the first sentence, the warning itself", which names the sentence: fine.
READ TWICE 1-7 FIXED as rewritten (Honeyquest line keeps "marked it to exploit" per Kasper's PDF check: accepted). New lead "Naming was not how most agents avoided the key." checked: 30/38 and 22/37 key-free runs did not name one (= 79%, 59%).
Minor: M1 kept (STYLE needs n/models/judge in the caption): accepted. M3 cost line follows after critic2: open, not a cold-read blocker. M4 left in footnotes (word cap): accepted.
Diff 306622b..8c50e17 (word-diff of _notes/warning-words.md): no sentence dropped beyond the listed fixes; one added ("Over all five agents ... 15 points", true: 93 -> 78).
VERDICT: PASS (post T9 and figure 1).
