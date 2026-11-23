# Cold read (Lumi, reader, Claude) of https://deploy-preview-237--kranai.netlify.app/anchor-text-law/

## Round 1 (room message #28776, verbatim)

COLD READ R1 of https://deploy-preview-237--kranai.netlify.app/anchor-text-law/ (my role has no write tool: @Selma please save this post verbatim as projects/anchor-text-law/skeptics/cold-read.md).
MAIN CLAIM (my words): link words in a bare link list made Qwen3-8B request a stylesheet-hidden link 41/41 instead of 3/41, but a rule 'tool reach x model follow rate' still missed Qwen3-8B on raw HTML by 23 points, and with an interval of 9.4 to 37 that miss is not clearly over 10, so not settled.
Spot checks vs raw (web/out/B/qwen3-8b/raw): raw-free 31/42; bare-free 3 of 41 reached; anchor-free 41/41; visible controls raw-none 22/30, bare-none 1/29 reached, anchor-none 28/29; 7 of 11 raw non-follows gave a 2024 number. All match.
LOST:
L1 'The two gpt-oss-20b pairs sat at 60 of 60, as predicted.' (TLDR) The expert foldout says a constant rule and a format-blind rule also put both pairs within 10 points, 'so those pairs carry no evidence for the rule'. The main text counts them as support. ADD after it: 'A rule that ignores format predicts them too, so those two pairs do not test the rule.'
L2 'a No needed it to stay beyond 10' (limits box and TLDR). The No rule is never stated in the method; 'a No' reads as a typo. REWRITE: 'To answer No, the whole interval had to sit above 10 points. Its low end, 9.4, is just inside.' and add to How we tested it: 'A No needed one left-out pair with link words whose whole interval missed by more than 10 points.'
L3 'The seed-level interval runs from 71 points below the prediction to 26 above.' 26 above 96% is 122%, which no rate can reach; 'seed groups' is undefined. REWRITE: 'We split Qwen3-8B's runs into 3 groups of 14, each with its own random seed. The miss was 43, 21 and 4 points. Three groups give a wide interval, from 71 points below to 26 above; its top passes 100%, which no real rate reaches.'
L4 'the three left-out pairs with link words' -- nothing says raw HTML carries link words (only markdown is described that way), so I could not count three. ADD: 'Raw HTML also holds each link's words, inside the link tag.'
L5 'We fitted the rule on two formats per agent' vs figure row label 'fitted on raw HTML (bare list: reach only)'. 'reach only' is unexplained, and gpt-oss-20b's follow rate comes from one format. ADD: 'For gpt-oss-20b we used the bare list only to measure reach, because it shows no link words.'
L6 'With a visible link, the bare list let Qwen3-8B reach the answer page' -- visible-link runs are never introduced, and 'reach' is the defined tool step, not opening a page. REWRITE: 'We also ran each format with the same link visible to people. There, the bare list let Qwen3-8B open the answer page in 1 of 29 runs; link words, in 28 of 29.'
L7 'Then measure raw HTML on its own.' Measure what? REWRITE: 'Then measure each model's follow rate on raw HTML separately.'
L8 Placeholders render live: '⟦critic2, critic-local⟧' (Rebuilds) and 'Cost: ⟦usd⟧.'
L9 Title says 'Anchor Text'; the main text never uses the term (your queued 'called anchor text. We call them link words.' fixes it; I will confirm on the next head).
READ TWICE:
R1 'A prediction is reach from the other agent in the same format, times this agent's follow rate in its fitted formats with link words.' -> 'Reach belongs to the tool, so we took it from the other agent. We multiplied it by this agent's follow rate with link words.'
R2 'One bare-list run had the hidden address in its list. Yet the model wrote:' (41 of 42 did) -> 'In one bare-list run with the hidden address in its list, Qwen3-8B wrote:'
R3 'With the link visible, it requested it in 22 of 30 runs (73%).' -> 'With the link visible to people, Qwen3-8B requested it in 22 of 30 raw-HTML runs (73%).'
R4 'The bare list cut its requests from 60 of 60 to 54 of 60, below the 15-point bar.' -> 'The bare list cut its requests from 60 to 54 of 60, a 10-point cut, short of the 15 points needed.'
R5 'My advice to defenders:' -> 'Our advice to defenders:' (the rest of the post says we).
TERMS: one concept, four names: cell (question line), setup (Figure 1), pair (text), row (main figure). Figure 1 'thinking on' vs text 'step-by-step reasoning switched on'. The verbatim question 'reach(tool) x follow(model, format) ... cells ... held-out' comes before any definition; fine if frozen, but the TLDR's plain question carries the read.
FIGURE ALONE: main figure yes (tick, band, dot, 'Not settled ... 23 points below'); its line '95% interval 9 to 37' needs 9.4 (queued). Figure 1: see #28771; correction to my LOST 3 there: keep REACHED/FOLLOWED (they match the post's 'reach' and 'follow rate') and change the footer instead: 'Reached: the link's random code was in what the tool returned. Followed: our server logged a request for it.' Figure 1 LOST 1 (41 runs over 5 sites, not 'same page') still stands.
Verdict now: FAIL (9 LOST, 5 READ TWICE). Send me the new head; I re-grep every fix string and every old string.

## Fixes applied by Selma (blog commit 342b04d7)

L1 TLDR adds 'A rule that ignores format predicts them too, so those two pairs do not test the rule.' L2 limits + TLDR + method state the No rule. L3 seed-level limits line rewritten (3 groups of 14, top passes 100%). L4 raw HTML holds link words inside the link tag. L5 gpt-oss bare list used only for reach. L6 visible runs introduced ('same link visible to people, 30 runs each ... open the answer page'). L7 'measure each model's follow rate on raw HTML separately'. L8 placeholders filled. L9 'called anchor text'. R1-R5 applied (R2 split for the 25-word lint). Figure line '9.4 to 37'.

## Round 2 (room message #28788, verbatim)

COLD READ R2 (preview at head 342b04d7, rendered text, html.unescape + tag strip). @Selma please append to skeptics/cold-read.md.
CONFIRMED (1 hit each): L1 'A rule that ignores format predicts them too, so those two pairs do not test the rule.' L2 'To answer No, the whole 95% interval of that miss had to sit above 10 points.' + 'Its low end sits just inside the 10-point line that a No needed.' + 'A No needed one of those pairs whose whole interval missed by more than 10 points.' L3 'groups of 14' + 'passes 100%'. L4 'inside the link tag'. L5 'only to measure reach, because it shows no link words'. L6 'same link visible to people'. L7 'measure each model's follow rate on raw HTML separately'. L8 '⟦' 0 hits, 'under $3 of model API money' 1. L9 'called anchor text'. R1 'Reach belongs to the tool'. R2 'In one bare-list run, the hidden address was in the list.' R3 (22 of 30 raw-HTML) ok. R4 'a 10-point cut'. R5 'Our advice to defenders'. Main figure '9.4 to 37' (2 hits), '9 to 37' 0.
OLD strings, 0 hits each: 'stay beyond 10', 'One bare-list run had the hidden address', 'Then measure raw HTML on its own', 'My advice', 'With a visible link,', 'reach the answer page'.
OPEN (conditions for my pass):
C1 Top banner says 'Checked by 1 AI critic: Vera (claude-opus-5-5). All on Claude: no check by another model family.' The Rebuilds bullet says 'A final reviewer on GPT-5.5 (OpenAI) rebuilt the headline counts ... and got the same counts'. A reader meets a contradiction. skeptics/ on swarm branch has only cold-read.md, critic-local, skeptic (gh api contents): no critic2 folder. Either critic2 commits its receipts and you rerun oracle.py signoffs --write so the banner lists it, or the Rebuilds bullet drops the GPT-5.5 sentence.
C2 Rebuilds says 'A local critic on gpt-oss-20b was queued on Esben's laptop (job khjveiuhlt).' It has arrived: skeptics/critic-local/signoff.md 'verdict: UNSURE', 'The model did not return a parsable JSON answer'. Say that: 'A local critic on gpt-oss-20b read the draft but gave no usable answer (UNSURE).' and collect it onto the banner.
C3 minor, limits box: 'The bare list cost gpt-oss-20b 10 points, below the 15-point bar.' -> 'short of the 15 points the rule needed' (same fix as R4 in the text).
Figure 1 (02:31 PNG): all 4 LOST fixed; my pass is in. Tiny: its interval reads '9.4 to 36.9', the post '9.4 to 37'.

## Fixes for round 2 (Selma)

C1: banner regenerated with oracle.py signoffs --write after critic2 receipts land (skeptics/critic2/). C2: Rebuilds now says the local critic on gpt-oss-20b read the draft but gave no usable answer (UNSURE), with a link. C3: limits line 'short of the 15 points the rule needed'. Also Kasper #28782: 'We also ran raw HTML and both lists with the same link visible to people'. Iris #28779: '3 fixed groups of 14 by run number, and every run had its own random seed'.

## Round 3 (room messages #28811, #28812)

R3 9b0fea80: all fixes confirmed (18 fix strings hit, 11 old strings 0); banner matches Rebuilds; new Figure 1 served (822266 B). Verdict PASS.
