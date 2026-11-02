# Deviations from PREREG.md

Each change from the frozen plan, with the reason. None of them changes the ranking rule or the
prediction.

- **D1. Classifier model.** PREREG.md names Claude Sonnet through `claude -p`. On November 2, 2026 the
  `claude` CLI was logged out on the laptop and on the Mac mini. The labels and the cluster reviews ran
  through omp's `completion` helper instead, which calls Claude Opus 5.5 on the same Claude subscription.
  Same prompts (classify.py, review.py). No paid API key was used. `classify.claude_p_submit` runs the same
  prompts through `claude -p --model sonnet`.
- **D2. Full text.** The forums' API returns plain text cut at 2,000 characters (51,176 of 79,990 posts and 65,227 of
  1,259,963 comments hit the cut). We found the cut after the first pass and fetched the full HTML of every post and
  of every cut comment (fetch.py `posts_html`, `comments_html`; fetch_ea.js likewise). Block quotes are
  removed from the HTML before the regex. The other comments are short enough that the plain text is the
  whole comment.
- **D3. Quote check.** PREREG.md drops a match whose sentence appears word for word in the parent comment
  or the post. The code first checked only the parent comment; it now checks both.
- **D4. Content filter.** The model refused a few snippets about biology (its content filter). A batch
  that fails is split into single snippets; a single snippet that still fails counts as "not an ask" and
  is marked `"refused": true` in data/labels.jsonl.
- **D5. Code run before the full corpus.** To test the code, cluster.py and review.py ran once on the
  part of the corpus that had downloaded (LessWrong comments from 2013 on). We read that dev ranking. After
  it, the merge threshold in review.py (cosine similarity at which two reviewed groups are offered to the
  model as one ask) moved from 0.85 to 0.80, because 0.85 offered only one pair. PREREG.md does not fix
  this number. The ranking rule and the prediction did not change.
- **D6. Askers counted per person, not per account.** PREREG.md says "distinct accounts". The precision
  and recall skeptics found that one person often has a LessWrong and an EA Forum account ("Jack_S" and
  "Jack_S🔸"), so a crossposted ask counted twice. Five of the first top 50 had one person only. We now
  count askers by display name in lowercase letters and digits (review.asker) and re-ranked the same
  LLM-reviewed groups without new LLM calls (review.rerank). Five asks left the top 50 and five entered;
  one more search agent checked the five new ones. The ranking rule's order is unchanged.
