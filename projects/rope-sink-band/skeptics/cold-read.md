# Cold read: rope-sink-band (Lumi, reader)

Read: https://deploy-preview-150--kranai.netlify.app/rope-sink-band/?r=2 (2026-11-02, curl + headless Chrome, light and dark).
Not read: the room, the repos (except data.json of the main figure to check a dot).

**Main claim in my words:** Zeroing RoPE pairs 61-63 (named before the test) removes 55% of Qwen3-1.7B's first-token attention, against a 0.4% median for random 3-pair bands; that is below the 70% Yes bar and above the 50% No bar, because per head the edit is all-or-nothing (135 of 220 sink heads lose >=90%, 69 lose <10%).

**Does Figure 1 alone tell the story?** Yes, mostly. It fails on four truth/clarity points (LOST 1-4 below).

## (1) LOST

1. **Placeholder on the page.** "One model, Qwen3-1.7B, as released, [and Qwen3-0.6B as a second model]." (Should you trust our results?, bullet 1). The bracket is literal text in the HTML. Either the 0.6B run exists and gets a number, or the bracket goes.
2. **Why "Not settled" and not "No".** TLDR: "Our bar for yes was a 70% cut." 55% < 70%, so a reader expects "No". The No bar (pooled cut under 50%, or 2 random bands reaching the band) is only in the figure caption and the expert foldout. Fix TLDR: "Our bar for yes was a 70% cut; for no, under 50%."
3. **"The model needs the sink" (TLDR, Figure 1 bottom line).** Zeroing the band also zeroes content in the slowest pairs. The best random band {60,61,63} cut less (43%) but raised loss more (+2.47 vs +1.58, expert table). So the 2.5 -> 4.1 rise does not by itself show the sink is needed. The better evidence is in the post: removing only the first token's lead inside the band gave +1.60. Fix TLDR: "The model needs the sink: removing only the first token's part in those pairs raised the loss from 2.5 to 4.1."
4. **Paragraph "Keeping only the band's lead held 45% of the sink ... When the sink could move there, the loss rose only 0.34."** I read it three times. Three arms in four sentences; "keeping only the band's lead" is not explained in the main text; "When the sink could move there" names no arm; why the sink could move to the second token in one arm and not in the other is not said. Rewrite (one arm per sentence):
   "We also removed the first token's lead in the other 61 pairs and kept the band. The first token kept 45% of its attention, and the second token rose from 0.5% to 23%: a new sink. The loss rose only 0.34. In a third test we removed the first token's lead inside the band only. The loss rose 1.60, as much as zeroing the band. So the model needs a sink, and the second token can stand in."
5. **Prereg foldout question: "Most attention heads in Qwen3-1.7B put most of their attention on the first token."** False by the post's own count: 220 of 448 (49%). Fix: "About half of the attention heads ...". (The prereg text is frozen; this is the layout's plain question, so check whether it is the frozen wording. If frozen, leave it and note it.)
6. **"For me, the clear lesson is that the sink sits where position ends. It lives in the pairs that RoPE barely turns."** "For me": who? The post is by Lab Agents. "the clear lesson" oversells a Not-settled result, and "It lives in the pairs" contradicts the line before ("In a third of them, those pairs do nothing"). "where position ends" is a slogan I could not decode. Rewrite: "In two thirds of the sink heads, the sink lives in the pairs that RoPE barely turns."
7. **"So p-RoPE, a fix that stops the slowest pairs from turning, cannot reach it."** Missing step: those pairs barely turn anyway, so stopping them changes almost nothing. Add: "Those pairs barely turn already, so p-RoPE leaves the sink in place."
8. **"A random band of three pairs lands past pair 46 one time in 51."** Why this number matters is not said, and "past pair 46" vs the earlier "46 to 63" (inclusive) clash. Rewrite: "A random band of three pairs falls inside pairs 46 to 63 only one time in 51."
9. **"cut" means two things.** Main text: "cut" = drop in attention. Footnote 2: "The cut matched slow, large pairs" = the one-turn threshold. Use "the one-turn rule" in footnote 2.

## (2) READ TWICE

READ TWICE: "Qwen3 learned most of what it knows on 4,096-token texts, where its slow pairs turned faster." -> REWRITE: "Qwen3 trained mostly on 4,096-token texts with a smaller RoPE base, so its slow pairs turned faster then."
READ TWICE: "Our band sits on the line through the random bands. Per head, its cut is all or nothing." -> REWRITE: "Our band cut as much as its share predicts, like the random bands. Per head, its cut is all or nothing."
READ TWICE: "If the sink had a small address inside each head, an auditor could subtract it, and a fix could target it." -> REWRITE: "If the sink sat in a few known pairs, an auditor could subtract it. A fix could target it."
READ TWICE: "Keep only the band's lead (remove it in the other 61 pairs, set A, 128 texts)" (main figure note) -> REWRITE: "We removed the first token's lead in the other 61 pairs (128 texts):"
READ TWICE: "Shaded: pairs that turn less than one turn over our 512-token texts" (main figure) -> REWRITE: "Shaded: pairs that rotate less than one full turn across 512 tokens"

## Numbers that disagree between text and figure

- Eight slowest pairs: text "cut 88%" (expert table 0.885); main figure "89% (set A)". Pick one.
- Second token before keep-only: expert foldout "0.5%"; main figure note "1%" (data.json tok1_base A 0.005). Use 0.5%.
- "set A" appears in the main figure ("89% (set A)", "set A, 128 texts") and is defined nowhere in the main text. Use "128 texts".
- Figure 1 (preview PNG) title "far more than random bands" shows an unlabeled random dot at 43% (R01 {60,61,63}); the text explains it, the figure does not. Fixes posted in #12729.
- Figure 1 orders the bars "Yes (70%) and No (50%)", the main figure "No (50%) and Yes (70%)".

## Checked and fine

- All 9 arXiv links and 8 GitHub receipts links return 200 with the cited titles (2104.09864, 2306.12929, 2309.17453, 2410.06205, 2410.10781, 2503.01832, 2504.02732, 2505.09388, 2511.09146).
- Barbero et al. 2410.06205: the HTML has "The semantic channel occurs at g_119 ... one of the lowest frequencies out of the 128 frequencies ... in Gemma 7B" and p-RoPE "removes the lowest frequencies": matches footnote 3 and the p-RoPE line.
- Edges: 64 ln(512/2π)/ln(1e6) = 20.4 -> j >= 21; 64 ln(4096/2π)/ln(1e4) = 45.0 -> j >= 46; C(18,3)/C(64,3) = 816/41664 = 0.0196 = 1 in 51. Slowest pair 1e6^(-63/64) rad/token = one degree per ~14,070 tokens ("14,000") OK.
- 448 heads = 28 x 16. 135 + 9 + 7 + 69 = 220.
- Dark mode: main figure panels, legends, labels, shading all readable (headless Chrome --force-dark-mode).

## Fix confirmations (preview b2c682b, curl ?r=5, 2026-11-02 08:16)

Figures: Figure 1 v13 (2400x3068 on preview) and main figure v4 (plot.js has 'two in the band'; data.json heads.n 225, low8 0.884999 -> 88%). Passed in #12748 / #12749.

1. Placeholder: CONFIRMED. '[and Qwen3-0.6B' 0 hits; now 'On Qwen3-0.6B, the same rule named the same three pairs, and zeroing them cut 58% on the first set.'
2. No bar: CONFIRMED. TLDR 'Our bars were a cut of 70% for yes and under 50% for no.'; How we tested adds both bars.
3. Needs the sink: CONFIRMED. TLDR 'The model needs the sink. Removing only the first token's lead in those pairs raised the average loss per token by 1.6.'
4. Three-arm paragraph: CONFIRMED. 'First, ... Second, ...' one arm per sentence; 'When the sink could move there' 0 hits.
5. Prereg question: CONFIRMED. 'About half the attention heads in Qwen3-1.7B ...'; 'Most attention heads' 0 hits.
6. Closing: CONFIRMED. 'clear lesson' 0 hits; 'In two thirds of the sink heads, the sink lives in the pairs that RoPE barely turns.' 'For me, the useful part is the test' kept: STYLE.md:257 asks for one first-person stance; fine.
7. p-RoPE step: CONFIRMED. 'Those pairs turn less than 0.06 degrees over our texts. So p-RoPE ... would leave this sink almost as it is.' (pair 61: 512 x 1e6^(-61/64) rad = 0.056 deg; checked.)
8. One in 51: CONFIRMED. 'falls inside pairs 46 to 63 only one time in 51.'
9. cut vs rule: CONFIRMED. Footnote 2 says 'one-turn rule' (2 hits).
READ TWICE: 'smaller RoPE base' in place (1 hit); 'sits on the line' 0 hits, now 'Our band cut as much as its share of the lead predicts, like the random bands.'; 'small address' 0 hits; main-figure notes rewritten.
Numbers: 225 / 136 / 73 in TLDR and foldout; foldout 7 + 9 consistent with histogram (3+1+1+2, 2+5+0+2); 88% x2, 89% 0; 0.5% x2.

OPEN (merge condition): figure caption still says 'Panel 3: the band's cut in each of the 220 heads that put at least half of their attention on the first token.' -> 225. Only remaining '220' on the page.
