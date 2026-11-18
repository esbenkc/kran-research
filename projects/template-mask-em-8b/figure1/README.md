# Figure 1 for template-mask-em-8b (Owain, prebuilt before the data)

`gen.py` builds `figure1.html` from the real generations and judge calls. It recomputes every rate with its own
code and asserts that it equals `out/analysis.json` (analysis.py). It picks the example by the rule frozen in
room post #24318, before any data:

- question = the broad question with the largest (full - base) misaligned rate on seed 0;
- sample = the lowest sample index where the seed-0 full-fine-tune answer is misaligned;
- every row shows its own answer to that same (seed 0, question, sample); the sampling key is paired across arms.

`text.json` holds every word of the figure. The kicker, head and subtitle are placeholders until the result is
known; `{name}` slots are filled with numbers by gen.py (`{full_broad}`, `{template_narrow}`, `{cut}`, ...).

## Five steps when all cells are judged

1. In the blog clone: `git pull`; the post must have `question:` in its front matter.
2. From this folder:
   `python3 gen.py --out ../out --analysis ../out/analysis.json --post <blog>/_notes/template-mask-em-8b.md --text text.json --dest <blog>/assets/figures/template-mask-em-8b-fig1/figure1.html`
   It prints the pick (question, sample, per-question gaps, verdicts) and every rate; check them.
3. Write the head (how the example looks, plain words, at most one number, not the post title), the kicker
   answer (Yes. / No. / Not settled.) and the subtitle in `text.json`; set `axis_max` just above the largest bar.
4. Render: from the blog root, `CHROME_PATH=... node _thumbs/figure1.mjs template-mask-em-8b-fig1`; open
   figure1.png and og.png.
5. Gate: `with-model-keys python3 tools/readq.py 8 <png>` (POST=<post file>) must pass at least 7 of 8, then
   commit figure1.html, figure1.png, og.png and `_data/figure1.json`, and set `figure1:` and `figure1_alt:` in the post.
