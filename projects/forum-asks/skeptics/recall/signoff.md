# Recall skeptic: how many real asks does the pipeline miss?

Checker: LabAsks.SkepticRecall. Date: November 2, 2026. I judged every sampled item myself against the
definition of an ask in PREREG.md. I did not call a model. Intervals are Wilson 95%.

## What I checked

1. **Classifier misses.** I took 150 random snippets that the classifier labelled "not an ask" (seed 11,
   from 22,199). Script: `sample_negatives.py`. Items: `neg_sample.jsonl`. My labels: `neg_judgements.jsonl`.
2. **Regex misses.** I re-ran filter.py's own text extraction, drops and sentence splitter on the full raw
   corpus. I kept the sentences with a test word (`\b(test|experiment|replicat|measure|study|would be
   curious|wonder whether)`) in eligible texts that have no snippet. Script: `scan_regex_misses.py`.
   Pool sizes: `pool_counts.json`. I read two samples:
   - 200 sentences drawn at random from the whole pool (seed 13): `regex_sample.jsonl` and `regex_judgements.jsonl`.
   - 200 more from the part of the pool that has an ask cue word, such as should, could, would, wonder,
     worth or "?" (seed 14). I used this sample to find the missed phrasings and to check the estimate:
     `regex_cue_sample.jsonl` and `regex_cue_judgements.jsonl`.
3. **Targeted check of the leaders.** I searched the same corpus for sentences that the regex did not
   catch and that match the topic of five top clusters (persuasion, talent programmes, METR RCT, Schelling
   steganography, natural abstractions; 2019 on). Script: `scan_topics.py`. Hits: `topic_hits.jsonl`. My
   labels: `topic_judgements.jsonl`. This test answers the sign-off question directly.

My scan reproduces the filter's eligibility counts. For example, it finds 48,558 eligible LessWrong posts,
which equals 54,918 minus the logged drops. Only 21 unmatched texts contain a regex hit, and in each of
those texts every hit was dropped as a quote. So the "no snippet" pool is the right pool.

## Numbers

**1. Classifier misses (ask=false snippets), n = 150**

- Clear asks: **2 of 150 = 1.3% (0.4% to 4.7%)**. Scaled to all 22,199 negatives: about 300 (80 to 1,050).
- Clear or borderline: 9 of 150 = 6.0% (3.2% to 11.0%).
- About AI: **0 of the 2 clear misses**. 4 of the 7 borderline items are about AI.
- The classifier is not the problem. Its misses are few, and most are vague requests that the
  definition allows it to exclude.

**2. Regex misses, n = 200 from a pool of 299,768 sentences (in 124,614 unmatched texts)**

- Clear asks: **2 of 200 = 1.0% (0.3% to 3.6%)**. Clear or borderline: 5 of 200 = 2.5% (1.1% to 5.7%).
- 18 of the 200 sentences are MathJax style-sheet text. filter.py's `html_to_text` keeps `<style>` content.
  This adds noise only and does not cause false matches.
- All 5 clear or borderline items fall in the cue part of the pool. The 154 sampled sentences without a cue
  word contain 0 asks (upper bound 2.4%).
- Cue sample, n = 200 from 71,560: **10 of 200 = 5.0% (2.7% to 9.0%)** clear asks, and **7 of those 10 are
  about AI**.

**Back-of-envelope (labelled estimate, not a measurement):** the regex missed about
299,768 × 1.0% ≈ **3,000 ask sentences (820 to 10,700)**. The cue sample gives a consistent figure:
71,560 × 5.0% ≈ **3,600 (1,960 to 6,400)**. For comparison, the regex found 5,819 asks. So the regex
catches roughly **60–65% of the asks that contain a test word**. For AI asks the figure is about 55%
(about 2,500 missed beside 2,922 found, using 7 of 10). This is still an upper bound on recall, because
the pool leaves out three things. First, asks with no test word at all ("try", "probe", "evaluate",
"studies"). Second, 63,972 unsampled test-word sentences in texts that do have a snippet. Third, texts
under 80 characters.

**3. Targeted check of the leaders (current ranking, after D6)**

For each cluster below, I found askers whose asks the regex never saw.

| Cluster (rank now) | Askers now | Missed askers I found (clear) | Borderline |
|---|---|---|---|
| Natural abstractions (5) | 3 | Leo Gao 2022, Nathan Helm-Burger 2024, Joe Kwon 2022 | none |
| Talent programmes (11) | 3 | NickLaing 2026, Samuel Holton 2024 | Ryan Kidd, NunoSempere |
| X-risk persuasion (1) | 4 | David_Moss 2023 (his other sentences sit in two separate groups) | Kabir Kumar, Max_Daniel |
| METR RCT (2) | 4 | none clear | elifland, titotal |
| Schelling collusion (8) | 3 | jake_mendel 2025 (Open Phil RFP) | none |

The search covered only these five clusters. The other clusters would gain askers too.

## Phrasings the regex missed most often

Counts are from the 12 clear misses in the two random samples. Examples from the targeted check are in
brackets.

1. **A modal with "you / one / we / they" as the subject (5).** Examples: "One could use a conditional place
   preference test", "you could invert the adversarial perturbation procedure", "We could try to measure",
   "Can we test this thing?" `someone_should` and `could_someone` accept only someone, anyone, people or
   labs. [Joe Kwon: "One could run useful empirical experiments"; NunoSempere: "we could run an RCT"]
2. **"We'd like to see", "we are interested in studies where", "I would be excited about people doing these
   experiments" (2).** `love_to_see` requires "I'd" or "I would" followed by "to see" or "to know".
   [Leo Gao: "We'd like to see work towards gathering evidence on whether natural abstractions holds"]
3. **"It would be great for someone to replicate" or "great to get GiveDirectly to test" (2).**
   `would_be_interesting` requires the verb right after "to".
4. **An imperative or "encourage" (2).** Examples: "Consider running it!" and "I'd strongly encourage people
   to try to replicate this". [NickLaing: "Running one MATS application round where you randomised entry
   ... would get you insanely good data"; Nathan Helm-Burger: "something we should seek to empirically
   measure"]
5. **Future-work wording with a singular "study" (1).** Example: "can be an interesting next study".
   `future_work` accepts "studies" but not "study". [Edward Cant: "A possible follow-up study could";
   Samuel Holton: a future-work list item]

## Items I disagree with

Classifier said "not an ask", but I judge them as asks:
- `ea-nZ7GZzrQjgdgCJynF-13`: "I'd love to see someone try this experiment". This is a pilot that tests whether
  a Transparent Idea Directory cuts transaction costs.
- `ea-f4MxxuKT7pzhuHmks-h56`: "Future studies could explore the long-term impact of various messaging
  strategies" on legislators. This is a concrete study.

Borderline items that the classifier labelled "no" (my reasons are in `neg_judgements.jsonl`):
`lw-YwAbQPhiyaMv8Nqk5-6`, `lw-sd3GMjLzBQtXHft5c-h9`, `ea-T5fSphiK6sQ6hyptX-h125`, `lw-keTc86gpcXw4CC6Jx-1`,
`lw-EJsABvCyaSDmPxmPt-0`, `lw-XaTTneKxCX6TGEiG8-7`, `lw-s7uD3tzHMvD868ehr-h8`.

The regex never saw these asks. They are listed above and in `topic_judgements.jsonl`:
- Leo Gao, EA comment `8gC2f6xfkJGWFFwW4`: test whether natural abstractions hold and how they scale with
  model size.
- Nathan Helm-Burger, LW comment `xdtazpwv5io26LNZL`: measure whether natural abstractions converge across agents.
- Joe Kwon, LW comment `5ncKoe8R8uMhJctXH`: run natural-abstraction experiments now, using human
  representation proxies.
- NickLaing, EA comments `GcEZQnC7nvTQJvXss` and `sjanxwybDyvP3sYJR`: randomise one MATS admission round.
- Samuel Holton, LW post `ewWbwHGeg7npEyTxw` (and its EA crosspost): measure MATS's counterfactual benefit.
- David_Moss, EA post `RYNtykh5xM467zRNj`: test which AI-risk messages work for which groups.
- jake_mendel, LW post `26SHhxK2yYQbh7ors`: assess whether LLMs converge on Schelling points.

Side finding, already fixed as D6: the same person with a LessWrong account and an EA Forum account
counted as two askers.

## Verdict

The classifier is fine: it misses about 1% of the snippets it sees. The regex is the weak step. It
catches about 60–65% of the asks that contain a test word, and fewer of the AI asks. Missed asks are not
rare at the top of the ranking. The leaders have only 3 or 4 askers, so 2 or 3 missed askers change the
order. With the clear misses above, natural abstractions would have 6 askers and lead the list.
Talent programmes (5, years 2024–2026) and persuasion (5) would pass the METR RCT (4). The set of
topics near the top stays about the same, but the published order, and the number-one ask in particular,
is not robust.

Two things would let me sign off:
(a) Add the five missed phrasing families to filter.py, then classify and cluster the new snippets. This
is a cheap change, because the classifier removes false matches.
(b) Or publish the counts as lower bounds ("at least N askers"), show the top as tiers, not as a strict
order, and state the measured regex recall.

SIGN-OFF: no, because the regex misses roughly a third or more of the asks, and in a targeted check
the missed asks add 2 or 3 askers to the leading clusters. That is enough to change which ask leads,
because natural abstractions would move from 3 askers to 6 and take first place.
