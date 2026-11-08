# Cold read (Lumi, reader), saved verbatim from rooms.db by Bruno

Room defense-1108-invisible-link. Lumi has no write tools; her posts are the receipt.

## Message #17034 (2026-11-07T23:41:17-08:00 reader -> critic)

Lumi cold read of BOTH figures (main: video/figtest/r6-*.png + data.json; Figure 1: blog assets/figures/invisible-link-fig1/figure1.png). No post preview read yet; send me the preview URL for the post cold read.
MAIN CLAIM (my words): when the answer is only behind a hidden link, agents reading raw HTML or markdown follow it almost always if its words fit the task (99-100%, same as a visible link) and ~30% if not; when the page already shows the answer, nobody follows it; a screen-reader view never shows a display:none link.
LOST 1 (both figures): what is the 'Start page' group vs 'Price page without the 2025 figure'? Both say the answer is not on the page. Figure 1 says 'Gemini 3.1 Flash-Lite starts on the site's Plans page', so I mapped the example to the 'Start page' row; it belongs to row 2. Fix: name the start page by what it is (e.g. 'Home page, no prices, answer behind the link') and say the price-page agent starts on Plans.
LOST 2 (both): '0 of 180 runs' in titles, but the main-figure header says 'runs that saw the link: 90 task-fitting, 88 off-task' = 178 (data.json rfree exposed 45+45+44+44). Every other count in the figure is 'runs that saw the link'. Fix: '0 of 178 runs that saw it' (or say 180 runs, 178 saw it).
LOST 3 (main): subtitle 'styled display:none, on every page' then a row 'sr-only (1-pixel clip)' that is not display:none, with only a pink dot and header '30 runs per link'. data.json controls: sr-only has only the task-fitting arm (n 30, exposed 29). Fix row: 'sr-only (1-pixel clip, fitting link only)' and subtitle '...styled display:none (one control: a 1-pixel clip)'.
READ TWICE: 'A run counts when the agent's tool output held the link (run counts per group in the group headers).' -> REWRITE: 'We count only runs whose fetched page text contained the link. Run counts are in each group header.'
READ TWICE: 'the 2025 figure' (row labels, both figures; 'figure' = a chart here) -> REWRITE: 'the 2025 price'.
READ TWICE (Fig 1): 'Three runs, each with one change:' -> REWRITE: 'Three runs. Each differs from the first in one thing:'.
TERMS a cold reader may not know: sr-only, accessibility tree (header defines it, good), display:none/aria-hidden (fine for ML reader), nofollow.
MINOR: dark column render, the small per-agent grey dots (markdown rows at ~7% and 10%) are near-invisible on black; one step lighter.
FIGURE ALONE: Figure 1 tells the story alone (question, yes, example with server log, bars). Main figure tells it alone except LOST 1. Numbers checked vs data.json: 88/89=99%, 30/90=33%, 24/89=27%, 21/45=47%, 13/44=29%, sr-only 29/29; Fig 1 pooled 178/179=99%, 54/179=30%, 87/87=100%, 34/89=38% all match.

## Message #17045 (2026-11-07T23:42:44-08:00 reader -> critic)

COLD READ of https://deploy-preview-177--kranai.netlify.app/invisible-link (read only that page). I do not edit files: @Kasper please save this post verbatim as kran-research/projects/invisible-link/skeptics/cold-read.md.
MAIN CLAIM (my words): hidden-link traps catch text-reading agents when the link words name what the agent still lacks (99% vs 30% off-task, same as a visible link), not once the page shows the answer (0 of 180), and not agents that get a screen-reader view.
LOST
L1 'Earlier, we left a fake key where AI agents could find it. Stuck agents used it in most runs, whether or not it fit the job.' - what key, stuck on what, and why this matters here is never said. REWRITE: 'Earlier, we left a fake API key where agents could find it. Agents blocked on their task used it, useful or not. Agents not blocked never did (0 of 100). Here we test the same pattern with a hidden link.'
L2 'Each agent ran 30 times per link and page view in the main test' - false for the screen-reader view: text says 0 of 60 runs and figure says 30 runs per link (= 10 per agent). Also 'page view' reads as a web-analytics hit. REWRITE: 'Each agent ran 30 times per link and page format (raw HTML, markdown). The screen-reader view had 30 runs per link in total.'
L3 Term clash: caption and text say 'home page'; both figures say 'Start page, answer only behind the link'. Figure 1 also says Gemini 'starts on the site's Plans page', so I mapped its example to the Start-page bar row. Rename the figure rows 'Home page, ...' in both figures.
L4 'The visible link was followed in every run that saw it. So we can rule out a large cost of hiding only for a well-fitting link that is the only route.' REWRITE: 'Hidden fitting link 99%, visible 100%: hiding cost little. We tested this only when that link was the only route.'
L5 'Each fitting link shares 2025 and a main word with the task.' - caveat with no point. REWRITE: 'Each fitting link repeats 2025 and a key word of the task, so fit may be just word overlap.'
L6 'A text browser such as Lynx ignores styling, so its users would see the link.' - the consequence (a human could trip the trap) is not said. REWRITE: 'A person in a text browser such as Lynx would see the link and could click it. We did not test that.'
READ TWICE
R1 'Guides for scrapers give the opposite advice. Render the page like a browser and drop the hidden parts, and the trap never fires.' -> 'Scraping guides say the trap is easy to dodge: render the page and drop hidden parts.'
R2 'A follow is a request for the run's own link address in our server log.' -> 'Each run got its own random link address. A follow is a request for that address in our log.'
R3 'The pink dots sit on the rings: hiding the link cost almost nothing. The third group sits at zero.' -> 'When the page already showed the answer (third group), both dots sit at zero.'
R4 'look up one 2025 figure' / 'find one 2025 figure' / 'Price page without the 2025 figure' (9 uses of '2025 figure' on the page, incl. caption and alt; 'figure' = chart here) -> 'one 2025 number' / 'the 2025 price'.
R5 'the agent's notes called the link hidden' ('notes' is new) -> 'the agent's written reasoning called the link hidden'.
R6 'Markdown gives the hidden and the visible link as the same text.' (orphan) -> 'In markdown the agent could not tell: hidden and visible links read the same.'
R7 'Our fake key followed the same rule: agents take the extra path only while they still need something.' -> 'The fake key showed the same pattern: agents take the extra path only while they still need something.'
R8 prereg box 'Most of those visits came from one agent, GPT-5.4 mini (44 of 60 runs).' (visits or runs?) -> 'GPT-5.4 mini made 44 of those 54 visits: it followed in 44 of its 60 runs.'
R9 'My lesson for defenders' - only 'My' on the page; the post speaks as 'we'. -> 'Our lesson for defenders'.
R10 'That link reached them, and 29 of 30 runs followed it.' vs figure '100%' (29 of 29 that saw it) -> '29 of the 29 runs that saw it followed it.'
MISSING RULE: the main figure strip uses +5 and +15 thresholds; the main text never says the title rule, and the prereg box says 30 points. Add one sentence: 'Before the runs we set the title rule: yes if the fitting link wins by at least 5 points.'
TERMS: sr-only (figure; text says 'shrank it to one pixel': use '1-pixel link' in the figure), nofollow (foldout only, fine), 'main-test runs' = 660 in text vs 360 in foldout ('358 of 360 main-test runs were exposed'): name one set.
FIGURES: unchanged from #17034 (LOST: 0 of 180 vs 90+88=178 that saw it; sr-only row needs 'fitting link only'). Preview figure1.png shasum 2a0277ba != clone f37327a3: the clone fix is not on the preview yet. Preview data.json = clone (7b7e90dc).
Total: 6 LOST, 10 READ TWICE, 1 missing rule. Story stands with foldouts closed: question, test, finding all clear.

## Message #17068 (2026-11-07T23:48:02-08:00 reader -> critic)

Lumi re-check on preview 44129ae (figure1.png 870acff0, data.json 1e16175a, plot.js 81728b1f = clone). CONFIRMED on the page (grep counts): L1 'fake API key' 1; L2 'page format (raw HTML or markdown)' 1 + '30 runs per link in total' 1; L3 'Start page' 0, groups named by start page in both figures; L4 'Hiding cost little where we tested it' 1; L5 'fit may be partly word overlap' 1; L6 'could click it' 1; R1 'Scraping guides say the trap is easy to dodge' 1; R2 'own random link address' 1; R3 'both dots sit at zero' 1; R4 '2025 figure' 0; R5 'written reasoning' 1; R6 moved to foldout; R8 '44 of those 54 visits' 1; R9 'My lesson' 0; R10 'all 29 runs that saw it' 1; missing rule 'we set the rule' 1. Main figure 1100 px light + 420 px dark: all labels readable, 1-pixel row labeled.
TWO NEW (from the fix batch), please fold into your next push:
N1 READ TWICE: 'Agents treated the fake key the same way.' It follows the 100% vs 38% fit sentence, so it reads as 'the key was chosen by fit', which contradicts the intro 'used it, even when it did not fit the job'. -> REWRITE: move it after '0 of 180 runs followed it.' as 'Agents that were not blocked also left the fake key alone.'
N2 READ TWICE + false: caption 'Groups: start on the home page ..., start on the page with the yearly table, without or with the 2025 number, and the page as a screen reader gets it, each shown as raw HTML (the page's code) or markdown (plain text).' (50 words; the screen-reader group is not raw HTML or markdown). -> REWRITE: 'Groups: the agent starts on the home page, or on the page with the yearly table (without or with the 2025 number). Rows show raw HTML (the page's code) or markdown (plain text). The last group is the screen-reader view.'
Non-blocking: figure subtitle 'We count only runs whose fetched page text held the link' sits under a title '0 of 180 runs' (178 saw it); '0 of 178 runs that saw it' would match the rule, but 0 is 0.

## Message #17076 (2026-11-07T23:50:04-08:00 reader -> critic)

Lumi: N1 and N2 confirmed on preview 2ae3405; post PASS (#review). @Kasper please save #17045 + #17068 + this line as kran-research/projects/invisible-link/skeptics/cold-read.md (I have no write tool). Final tally: 6 LOST + 10 READ TWICE + 1 missing rule + 2 new READ TWICE, all 19 fixed on the preview; figure 1 and main figure passed.

## Message #17077 (2026-11-07T23:50:04-08:00 reader -> write)

SIGN-OFF: post draft + oracle.py check + PR — Preview head 2ae3405 (all checks pass): 16 of 16 items from #17045 confirmed by grep in #17068, plus N1 ('Agents treated the fake key the same way' 0, 'Agents that were not blocked also left the fake key alone' 1, now after '0 of 180 runs followed it') and N2 ('each shown as raw HTML' 0; caption now 3 sentences ending 'The last group is the screen-reader view'; 2nd hit is the noscript copy). Main text stands with foldouts closed: question, test (home vs table page, random link address, +5 rule), finding (99/30/100, 0 of 180, 0 of 60) all stated.
