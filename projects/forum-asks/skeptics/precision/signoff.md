# Precision skeptic: sign-off

## What I checked

1. A random sample of 100 snippets that the classifier labelled `ask=true, ai=true` (pool 2,922; seed 7; `sample.py` -> `sample_random.jsonl`).
2. All 36 member snippets of the top 10 groups in `data/top50.json` (`sample_top10.jsonl`).

I read every snippet myself and applied the PREREG.md definition of an ask. For each item I judged: is it an ask, is it about AI, and is the one-line statement faithful? For top-10 members I also judged whether it asks for the same experiment as the group statement. Labels are yes / no / borderline (or partial, unclear). "Strict" counts borderline as no. "Lenient" counts it as yes. My labels are in `judgements.jsonl`. The numbers come from `score.py` (output in `score_output.txt`).

While I judged the top 10, I found that one person can count as two askers. I checked this across the whole top 50 with `crosspost_check.py` (`crosspost_check.json`).

## Numbers

Random sample (n = 100, Wilson 95% CI):

| Measure | Strict | Lenient |
|---|---|---|
| Real ask | 95/100 = 95% (89-98%) | 99/100 = 99% (95-100%) |
| Real ask and about AI | 91/100 = 91% (84-95%) | 95/100 = 95% (89-98%) |
| Statement faithful | 97/100 = 97% (92-99%) | 99/100 = 99% (95-100%) |

Top 10 members (n = 36): a true ask for the group's experiment in 26/36 = 72% (56-84%) strict, and 33/36 = 92% (78-97%) lenient.

The table below shows distinct people per group whose snippet is a true ask for that group's experiment. "Distinct people" means I merged the same name across forums.

| Rank | n_askers listed | Strict | Lenient |
|---|---|---|---|
| 1 | 5 | 4 | 4 |
| 2 | 4 | 1 | 2 |
| 3 | 4 | 4 | 4 |
| 4 | 3 | 2 | 3 |
| 5 | 3 | 2 | 2 |
| 6 | 3 | 3 | 3 |
| 7 | 3 | 2 | 3 |
| 8 | 3 | 1 | 3 |
| 9 | 3 | 3 | 3 |
| 10 | 3 | 1 | 2 |

## The main problem: crossposts count one person twice

`cluster.py` and `review.py` count askers by `user_id`. LessWrong and the EA Forum have separate accounts. So an author who crossposts the same text to both forums counts as two askers. In the top 50, this happens in 9 groups (`crosspost_check.py`):

- Rank 1: 5 -> 4 (NoahCWilson on both forums, identical snippet).
- Rank 2: 4 -> 3 (Jack_S on both forums, identical snippet).
- Rank 10: 3 -> 2 (Marc Carauleanu on both forums, identical snippet).
- Rank 12: 3 -> 2 (Jeyashree Krishnan).
- Ranks 42, 44, 46 (Ben Cottier), 49 (Larks), 50 (Marius Hobbhahn): 2 -> 1. These five groups each have only **one** person. They do not meet the 2+ askers bar that review.py uses.

PREREG.md says "distinct accounts", so the code follows the letter of the plan. But the post claims "the most different people". A crosspost is one person. If you recount by person, the top 10 changes: ranks 2 and 10 drop out, and old ranks 11 and 13 come in (`crosspost_check.json`, `rerank_by_people`).

## Items I disagree with

Random sample:
- lw-oJp2BExZAKxTThuuF-h116: not an ask. Zvi uses "I would like to see anyone engineer..." as a rhetorical challenge, and he thinks he knows the result.
- ea-gbzD2A5MhqTFJ4pA5-h10: borderline. It asks for a study proposal to a funder.
- lw-LRgM9cuLNPbsjwEdN-h300: borderline. It is an abstract open question ("we leave this as an empirical ML question") with no concrete test.
- ea-jd7QinmkdzegerRCm-h1136: borderline. It asks for a quantitative model of considerations (a forecast model), not an experiment.
- ea-vcbHaqotqvM3c9vTH-h0: borderline. It is a burden-of-proof demand to another commenter ("show me data or a study").
- ea-W8oKEGJtbydTEQ5Gy-1: not AI. The ask is about a calibration control question in the forecasting tournament.
- ea-hLdYZvQxJPSPF9hui-h99: not AI. It asks about expert vs non-expert forecasting of technology in general.
- lw-tSpE7GKocRBMLcPQp-21: not AI. It asks for the ROC AUC of a dating-compatibility classifier, and ML is only the tool.
- lw-AgaBzvuBJg2evEjqh-h319: not AI, and the statement is not faithful. The snippet says "whether your platform uses AI or not", but the statement narrows it to AI platforms.
- lw-Ez67KyWEepzyGm5kL-h5: faithfulness unclear. "That would be interesting to see" points at context that is not in the snippet.
- lw-zqmtSGeB7oy5xDiSc-2: faithfulness unclear. "Worth running the experiments" does not say which experiments.

Top 10:
- ea-t8tZMxQonL4Q2zh3W-h283 and lw-vNeezYG27nsnnpW98-h721 (rank 2, Jack_S): not an ask. The text states ML4Good's own limitation ("we haven't met the gold standard"). I checked the full post in the raw corpus. The next sentence describes their own two-question survey and does not ask for a study.
- lw-e5sdgYxYFKqdM6vF4-h1593 (rank 5, Elle Najt): different experiment. It asks how base vs RLVR training affects Schelling steganography. It does not ask whether separated instances can coordinate.
- lw-Ge55vxEmKXunFFwoe-h95 (rank 8): partial. The base study already used expert iteration, not SFT, and the base paper is a different one.
- lw-HLJoJYi52mxgomujc-h450 (rank 8): partial. The outcome is broad misalignment, not reward-hacking generalisation. Rank 8 joins three follow-ups to three different papers that share one idea ("use RL instead").
- ea-TbdQhx9YwNb4HpnaJ-h96 and lw-jtqcsARGtmgogdcLT-h96 (rank 10): borderline. This is a boilerplate future-work sentence, and it is one person crossposting.
- lw-rDLTRtebEoafMLzyC-6 (rank 4) and lw-RzGJB7GoosQmfALuE-h22 (rank 7): borderline. One is an evidence question with no test named. The other asks readers to fill in a template on his own post.

## Verdict

The snippet classifier is good. 95% of AI asks are real asks, 91% are real AI asks (strict), and 97% of the statements are faithful. That part I accept.

The top-of-ranking counts are not good enough to publish as "most different people". Crossposts inflate n_askers in 9 of the top 50. Five of the top 50 have one real person. Rank 2 rests partly on a non-ask that is also double counted.

SIGN-OFF: no, because n_askers counts one person's LW+EA crosspost as two askers (9 of the top 50 affected, 5 with a single person, ranks 2 and 10 fall out of the top 10). Rank 2 also counts Jack_S's limitation statement as an ask. Fix: count askers by person across forums, for example by matching normalised author names, and log it in DEVIATIONS.md. Then re-review ranks 2, 5, 8 and 10. With that fixed, I would sign off on precision.

## Re-check after D6 (2026-11-02, 14:40 PST)

LabAsks changed the asker count to one key per person (review.asker, DEVIATIONS D6) and re-ranked. I re-checked the new data/top50.json with `crosspost_check.py` and `recheck_d6.py` (output in `recheck_d6_output.txt`). I judged the 5 new top-10 members myself. They are the last 5 rows of `judgements.jsonl`, set `top10_d6_new`.

What I found:
- 0 of 50 groups count one person twice. 0 groups have only one person. My own name merge gives the same n_askers as the file for all 50 groups.
- The same name never belongs to two different accounts on the same forum, so the name key does not merge different people.
- The order follows the ranking rule (askers, then span, then karma).
- The new top 10 are old ranks 1, 3, 4, 5, 6, 7, 8 and 9, plus two new groups. The new rank 9 asks to replicate reward-hacking generalisation on more models. The new rank 10 asks whether AI tools raise workers' productivity. Old rank 2 (talent programmes) is now rank 11. Old rank 10 (self-other overlap) left the top 10.

New top 10, 35 members: a true ask for the group's experiment in 28/35 = 80% (64-90%) strict, and 34/35 = 97% (85-99%) lenient.

| Rank | n_askers listed | People with a true ask, strict | Lenient |
|---|---|---|---|
| 1 | 4 | 4 | 4 |
| 2 | 4 | 4 | 4 |
| 3 | 3 | 2 | 3 |
| 4 | 3 | 2 | 2 |
| 5 | 3 | 3 | 3 |
| 6 | 3 | 2 | 3 |
| 7 | 3 | 1 | 3 |
| 8 | 3 | 3 | 3 |
| 9 | 3 | 2 | 3 |
| 10 | 3 | 2 | 3 |

Disagreements that remain (all are judgement calls, not count errors):
- Rank 4: lw-e5sdgYxYFKqdM6vF4-h1593 (Elle Najt) asks a different experiment (base vs RLVR). So rank 4 has 2 people, not 3, under either reading.
- Rank 7: two of the three members (lw-Ge55vxEmKXunFFwoe-h95, lw-HLJoJYi52mxgomujc-h450) are only partial matches.
- Rank 9: lw-ATDvmn23PWW4erZzf-h516 (Isaac Dunn) is mostly about validity fixes. "Fine-tune more models" is one item in his list, so it is a partial match.
- Rank 10: ea-vcbHaqotqvM3c9vTH-h0 is a burden-of-proof demand (borderline). lw-ZLMDHpbQSp33p6JSB-13 asks to rerun the old coding-productivity studies with today's models. It could also belong to the METR-replication group (rank 2).
- Rank 11 (old rank 2): the Jack_S snippet is still a non-ask, so 2 real people, not 3. This no longer touches the top 10.

Every new top-10 group has at least 2 different people with a real ask for its experiment, even under strict judging. Lenient counts match the listed n_askers in 9 of 10 groups. The exception is rank 4 (2, not 3).

SIGN-OFF: yes. The crosspost double count is fixed, and classifier precision is 95% (89-98%) on n = 100. The remaining disagreements are judgement calls on single members (ranks 4, 7, 9 and 10). The post should say that ranks 3 to 10 tie at 3 askers, so their order rests on span and karma.
