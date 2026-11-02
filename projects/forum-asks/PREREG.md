# Pre-registration: forum-asks

Fixed: November 2, 2026, 13:25 PST, while the corpus was still downloading. No ask had been
classified, clustered or ranked yet. This file is the method and the prediction. Any later change
goes to DEVIATIONS.md with a reason.

## Question

Do the experiments that LessWrong, the AI Alignment Forum and the EA Forum ask for most often, by the
most different people, mostly stay untested?

## Corpus

Every post and comment that the public GraphQL APIs of lesswrong.com (which also serves the Alignment
Forum, flag `af`) and forum.effectivealtruism.org return, from the first post to November 2, 2026.
Deleted texts and texts with negative karma are dropped.

## What counts as an ask

A sentence in a post or comment that asks for a concrete experiment, measurement, replication,
analysis of existing data, or proof that somebody could run or write, where the writer does not know
the result. Examples that count: "someone should test whether RLHF models sandbag on X", "I'd love to
see this replicated on a bigger model", "has anyone tried training a probe on Y?". Examples that do not
count: requests for a post, an explainer, an opinion, funding, a tool without a test, a rhetorical
question, and a question about a fact that is already known.

## Filter (cheap first)

1. Regex for ask phrases, tuned for recall (filter.py: "someone should test/try/check/run", "I'd love to
   see", "has anyone tried/tested", "would be interesting to test/see/check", "open question", "hasn't
   been replicated", "I wish someone", "an experiment that", and similar).
2. Drop texts under 80 characters, deleted texts, karma below 0, and a match whose sentence appears word
   for word in the parent comment or the post (a quote).
3. Claude Sonnet (subscription, `claude -p --model sonnet`) reads each surviving snippet in batches of
   about 40 and answers in JSON: is it an ask (as defined above)? If yes, a one-line experiment
   statement, a field tag, and whether it concerns AI or AI alignment.

## Clustering

Local sentence embeddings of the one-line statements, then agglomerative clustering on cosine
distance, so one ask repeated over the years becomes one cluster. Claude Sonnet then checks every
cluster that could enter the top 50 and splits clusters that merge different experiments. Each final
cluster gets one canonical statement.

## Ranking rule

Only asks that concern AI or AI alignment enter the ranking (all three forums). Clusters are sorted by:

1. the number of distinct askers (distinct accounts), most first;
2. then the span of years (last ask year minus first ask year), longest first;
3. then the summed karma of the asking posts and comments, highest first.

The top 50 are published with links to every original post or comment.

## Open or answered

For each of the top 50, one search (web and the three forums) for a published test of the asked
question. "Answered" needs a link to a paper, post or dataset that ran the asked test or proved the
asked result. Everything else is "open, as far as we found". A partial test (a smaller model, one
setting) counts as answered only if the asker's question is answered for the setting they named.

## Prediction

At least 25 of the top 50 asks stay open (no clear published answer found).
Confidence: 60%.

- If right: the community's most repeated test requests mostly go untested, so a list of them is a
  ready research agenda for cheap agent runs.
- If wrong: most repeated asks do get answered, and the list mainly shows what the field already
  settled.

The title answer is yes (CONFIRMED) if at least 25 of the top 50 are open, no (REJECTED) otherwise.
