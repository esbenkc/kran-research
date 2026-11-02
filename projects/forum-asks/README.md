# forum-asks

Which tests do LessWrong, the AI Alignment Forum and the EA Forum ask for most often, by the most
different people, and are they still untested? Frozen plan: `PREREG.md`; changes: `DEVIATIONS.md`.
Post: https://blog.kran.ai/forum-asks/

## Corpus (fetched November 2, 2026, public GraphQL APIs)

| forum | posts | comments | first comment |
|---|---|---|---|
| LessWrong (incl. 4,696 AF posts, 25,839 AF comments) | 54,918 | 1,090,763 | 2006-11-22 (Overcoming Bias imports) |
| EA Forum | 25,072 | 169,200 | 2013-10-14 |

The API's plain text stops at 2,000 characters, so every post and every cut comment was fetched again
as full HTML (DEVIATIONS D2). The raw cache is 3.1 GB and is not committed; `data/raw_sample/` holds one
page of each kind. `fetch.py` (LessWrong, plain HTTP) and `fetch_ea.js` (EA Forum, from a headless
browser page, because a Vercel bot check blocks plain HTTP) rebuild it, with 1 s between calls.

## Pipeline

1. `filter.py`: regex for ask phrases (17 patterns), drops deleted texts, karma below 0, texts under 80
   characters, events and quotes -> `data/snippets.jsonl` (28,018 snippets from 24,559 texts;
   committed as `snippets.jsonl.gz`, `gunzip` it first) and `data/corpus_stats.json`.
2. `classify.py`: Claude labels each snippet (ask or not, a one-line statement, a field, AI or not), 40 per
   call -> `data/labels.jsonl`. 5,819 asks, 2,922 about AI. 5 snippets refused by the model's content filter.
3. `cluster.py`: bge-base-en-v1.5 embeddings of the AI statements, agglomerative clustering (average
   linkage, cosine distance 0.30) -> `data/asks.jsonl`, `data/clusters.jsonl` (1,112 clusters).
4. `review.py`: Claude splits each cluster with 2+ askers into groups that ask for the same experiment,
   then checks near-identical groups for merges -> `data/groups.jsonl` (1,847 groups), `data/top50.json`.
5. `data/answered.json`: five search agents looked for a published test of each top-50 ask (rule in
   PREREG.md; every "answered" link was opened).
6. `build.py`: the figure data, `data/table.md` and `data/top50.csv` (every asking text with its link).

LLM calls ran on the Claude subscription (omp's `completion` helper, Claude Opus 5.5; DEVIATIONS D1).
Model-API spend: $0.

## Rerun

```
python3 fetch.py lw comments; python3 fetch.py lw posts; python3 fetch.py lw posts_html; python3 fetch.py lw comments_html
CHROME=... bun fetch_ea.js posts; ... comments; ... posts_html; ... comments_html
python3 filter.py
python3 -c "from classify import run, claude_p_submit; run(claude_p_submit)"
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python sentence-transformers scikit-learn numpy
.venv/bin/python cluster.py
# review.run(submit, embed) from Python, embed = cluster.py --embed
python3 build.py <figure folder>
```

`skeptics/`: independent checks of precision, recall and the ranking, each with `signoff.md`.
