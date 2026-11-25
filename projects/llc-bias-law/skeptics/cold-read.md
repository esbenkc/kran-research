# Cold read: llc-bias-law (Lumi, reader)

Read: https://deploy-preview-19--athanorscience.netlify.app/llc-bias-law/ (fetched 2026-11-25 07:36 PST), Figure 1 PNG (figure1.png?v=96b042c500dd14df), main figure in light and dark theme (Playwright screenshots). Room not read.

Main claim, in my words: a two-part formula (an exact term for the bowl-shaped directions plus one fitted curve in the spring looseness q) predicted the SGLD error on five unseen network shapes within 5 points in 82% of 180 settings, but a lookup of the same setting on a training shape did almost as well (median miss 2.0 vs 1.9), so the answer is Not settled; the spring sets most of the error.

Main text with foldouts closed: 881 words (incl. figure captions).

## LOST

L1. Top box: “The split curve did not beat a lookup of the training shapes at the same setting” -- split curve and lookup are used ~400 words before either is defined (TLDR / figure caption). Fix: define both inline here.
L2. Top box: “Counted against the curve only, two simpler rules would beat it, and the answer would be no.” -- I cannot tell what is counted against the curve. REWRITE: “If unsettled walks counted as misses for the curve alone, two simpler rules would beat it.”
L3. Top box: “We scored settled error only, so the curve says nothing about the extra error of short walks.” -- “settled error” clashes with “settled settings” (walks whose halves agree). REWRITE: “We scored only the long-run error, so the curve says nothing about short walks.”
L4. “We fit it in one number, q = nβ/(2γ²): how loose the spring is, compared with the pull of the loss.” -- reads as a one-parameter fit, but the prereg foldout says “a law with two fitted numbers”. REWRITE: “The fitted curve depends on one number, q = nβ/(2γ²): the spring's looseness against the loss's pull.”
L5. Unit of “points”: readings are learning coefficients (227.5) and errors are percents. “agree within 2 points” (settled rule, How we tested) and “agree within 1 point” (walk-length rule) read as learning-coefficient units. Fix: say once “percentage points of the true value”.
L6. “To keep the spring's error under 5%, keep q above about 5, so the spring below about √(nβ/10). At an inverse temperature of 100, that is about 3 on the rank-1 training shape. The rank-8 shape tolerates 4.4.” -- the formula has no shape in it, so why do rank 1 and rank 8 differ? Say in one sentence why (I assume the bowl part differs), or say q>5 is the cautious rule for every shape. “at the smallest step” is also unexplained.
L7. “Take the spring that came closest in our last LLC post's test. With it, a perfect sampler on noise-free data reads 30% to 33% low, while that post's test allowed 10%.” -- “perfect sampler” is undefined here (later: exact sampler, Gibbs) and I do not know what “that post's test” was. REWRITE: “Our last LLC post allowed 10% error against a known answer. At its best spring, even an exact sampler reads 30% to 33% low.”
L8. Title “WORKING TITLE: A Spring-Strength Law for SGLD ...” -- still a working title, and “Law” claims more than Not settled.
L9. The “We asked” line (“does one curve in nbeta/gamma^2 predict ... at held-out ranks and widths”) is the first thing a reader sees: raw ASCII math and “regular and singular parts” before any definition. It is the verbatim prereg question, so I only flag it; the prereg foldout version is the readable one.

## READ TWICE

READ TWICE: “The exact sampler put a median of 5.4 points on the spring and 0.6 points on the steps.” -> REWRITE: “The spring alone gave a median error of 5.4 points. SGLD's steps changed it by a median 0.6.”
READ TWICE: “An exact sampler without steps (Gibbs sampling) gave each setting's error from the spring alone.” -> REWRITE: “Gibbs sampling, an exact sampler with no step size, measured the error from the spring alone.”
READ TWICE: “The lookup missed by a median of 2.0 points, against the curve's 1.9, and we required half.” -> REWRITE: “The curve's median miss was 1.9 points, the lookup's 2.0. We required half the lookup's.”
READ TWICE: “So our pre-registered prediction failed on its second half.” (the second half is never stated in the main text) -> REWRITE: “So our prediction failed its second part: half the median miss of every simpler rule.”
READ TWICE: “In this article, we ask: once the bowl part is taken out, does one curve in q predict the sampler's error on new network shapes?” (25 words) -> REWRITE: “We ask: with the bowl part removed, does one curve in q predict the error on new shapes?”
READ TWICE: “The curve gives a cautious spring rule at the smallest step.” -> REWRITE: “At the smallest step size, the curve gives a cautious rule for the spring.”
READ TWICE: “On the right, the five shapes fall close to one curve in q, and they spread most at the tightest spring.” -> REWRITE: “On the right, the five shapes follow one curve in q. At the tightest spring their means differ by 10 points.” (data.json: shape-mean spread of measured error at q=0.1 is 9.9 points; 3.8 or less at q=10 and 100)

## Terms I did not know (main text)

Gibbs sampling (named, not explained); rank of the target map; percentage points of an error that is itself a percent; perfect vs exact sampler vs Gibbs (three names, one thing); bowl part / regular directions / bowl-shaped part, singular part / flat parts (prereg) -- pick one pair; inner width / inner units / hidden units (Figure 1). Foldout only: special point w*, another branch of Aoyagi's formula, quartic coefficient.

## Figures

Figure 1 alone tells the story: question, answer (Not settled), one unseen shape (227.5 = 94 + 133.5), curve vs lookup vs measured at four springs, and the bars with the needed 1.0. Its numbers match the text (82%, 180, 1.9, 2.0, 2.2, 4.6, 7.4). Small: bar label “one constant” is “Average training error” in the expert table; “the number we first planned” is undefined in the figure alone.
Main figure alone says Yes: 148 of 180 in the band, solid lines on the dots. The lookup, which decides Not settled, is absent. Fix: one caption line, e.g. “A lookup (same setting, training shape) did almost as well: median miss 2.0 vs the curve's 1.9.” Also the rank-5 pink and rank-4 magenta are hard to tell apart in both themes. Dark mode: all labels, dots, crosses, lines readable.

## Fix confirmations
Head afed809 (blog), preview fetched 07:45 PST. Source: 13 fix strings present (1 each), 10 old strings at 0. Preview: “required half the lookup” 1, caption “A lookup of the same setting on a training shape had a median miss” 1, “We fit it in one number” 0.
- L1-L7: fixed as proposed. L6 now says why (“more of its LLC sits in the bowl part”). L5 partly: a units sentence added; “agree within 2 points” follows it, fine.
- All 7 READ TWICE: applied.
- Main figure: lookup 2.0 vs 1.9 is now in the post caption under the figure.
- Open L8: title still “WORKING TITLE ... Law” (6 hits in preview), naming step pending.
New on this head:
READ TWICE: “Applied to our network, the deep-linear-network settings of Lau and colleagues read 2.4% and 3.7% high.” -> REWRITE: “We ran the settings of Lau and colleagues on two of our shapes. The walks read 2.4% and 3.7% high.”
READ TWICE: (limits) “The 29 unsettled settings, about as many as walk noise alone would flag, count as misses for every rule.” -> REWRITE: “The 29 unsettled settings count as misses for every rule. Walk noise alone would flag about that many.”

Head 5ab84cc (blog source; preview stalled at b454d3e, PR head not synced): title set by naming step (“One Curve Predicts SGLD's LLC Error in Reduced-Rank Regression, Not Clearly Better Than Lookup”, WORKING TITLE 0 hits). Both round-2 rewrites present (1 hit each), old strings 0 (“Applied to our network”, “about as many as walk noise”). Shape numbers changed to 5.0 points / 93% (4.5 / 94% now 0 hits in the post; Figure 1 html has neither). Main figure carries the lookup line (plot.js intro + left note, Playwright light+dark). L1-L9 closed; L9 (verbatim question) kept by rule. Title word “Lookup”: Kasper keeps it (defined in the line under the title).
Verdict: PASS. To confirm on the live page after merge: title, the two round-2 sentences, 5.0/93%.

Preview synced (PR head 3260a2d), rendered page checked: title “One Curve Predicts ...” 4 hits, WORKING TITLE 0; both round-2 sentences 1 hit each; 5.0 points / 93% present, 4.5 / 94% 0; old strings (“We fit it in one number”, “perfect sampler”, “Counted against the curve only”, “Applied to our network”) 0. Cold read confirmed on the deploy preview.
