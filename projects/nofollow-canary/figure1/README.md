# Figure 1 (Owain) — nofollow-canary

Figure 1 sits above the TLDR of `_notes/nofollow-canary.md` and is the post's thumbnail. It shows ONE real example
pair and the run's counts. Every number and every quoted line comes from the job's output, never typed by hand.

    bash figure1/render.sh [BLOG_CLONE]     # gen.py, then the blog's own renderer (figure1.png, og.png, _data entry)

- `gen.py` reads `results/analysis.json` (from `analysis.py`), `web/out/qwen3-8b/raw/<arm>-free/<NNN>.json` and
  `web/out/qwen3-8b/access.log`. It recounts every arm's follows from the raw files and the log and stops if they
  differ from `analysis.json`. It writes `<blog>/assets/figures/nofollow-canary-fig1/figure1.html` and
  `figure1/sources.json` (the example's files, the picked index, the counts and `alt`: the `figure1_alt` text for
  the front matter, double-quote safe).
- `text.json`: the head, subtitle and answer line for each status, with `{name}` slots that gen.py fills from the
  counts. gen.py stops on a `[FILL]` left in the text it uses.
- Without `results/analysis.json` gen.py draws the PENDING scaffold: the left card is the parent run's real Qwen3-8B
  raw-HTML miss (`parent_example.json`, extracted from `origin/swarm/defense-1123-anchor-text-law`,
  `projects/anchor-text-law/web/out/B/qwen3-8b/raw/raw-free/035.json`), and the counts read "waiting for the laptop run".
  The scaffold is for the draft PR only and is never published.
- `mock_fixture.py` builds fake job outputs (parent Qwen3-8B transcripts with fresh ids, follow counts set per
  scenario) to test the landed mode. Mock trees live in /tmp only; nothing from them is committed.

## Example-pick rule (frozen 2026-11-23 23:40 PST, before any run of this room's job)

The example is ONE run index i, shown in two arms: raw HTML with the tag (`raw`, left card) and raw HTML without it
(`rawnf`, right card). The same i gives the same site recipe, task wording, link order and sampling seed in both
arms (PREREG s1, s2), so only the attribute differs.

1. Kind of pair, from the status in `results/analysis.json`:
   - CONFIRMED: left misses, right follows.
   - REJECTED: left misses, right misses.
   - UNCONFIRMED: left misses, right follows if rawnf's follows minus raw's follows is at least 9 runs (10 points
     at n = 90); else left misses, right misses.
   (follow = the access log holds a GET of the run's canary id; miss = it does not.)
2. Candidates: i in 0..89 where both runs reached the canary (its id is in at least one observation) and every
   missing run of the pair names the canary in its reasoning (the canary id or the link's anchor text appears in a
   reply after the first observation that holds the id), so the card can quote what the model read.
3. Pick the lowest candidate i whose template is `pricing` (the Northwind Cloud price task, i mod 5 = 0); if none,
   the lowest candidate i of any template; if none, drop condition 2's naming clause and repeat; if still none, the
   lowest i with a left miss and both runs reached, and the right card shows what that run did.

Quotes (fixed now): for a miss, the first sentence that names the canary (id or anchor text, case-insensitive) in
the replies after the canary was first read, plus the sentence after it, cut at 45 words with an ellipsis; for a
follow, the `thought` field of the reply that requested the canary id. The next-step line is the first logged GET
of that run after its start page. The answer line is the run's final answer, cut at 80 characters, marked
"last year's price" when it holds the template's 2024 figure and "the 2025 price" when the scorer marked it correct.

The markdown pair (mdhtml, mdnf) and plain markdown (md) appear only as counts. mdpad8 (length dose) is not in
Figure 1; it lives in the main figure.
