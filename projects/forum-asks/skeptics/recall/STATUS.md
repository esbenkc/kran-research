# Recall skeptic: stopped early

The recall skeptic stopped on a provider error (HTTP 400 from the model API) before it wrote signoff.md.
Its last progress report, copied here by the writer (LabAsks):

- Classifier misses: of 150 random snippets that Claude labelled "not an ask" (seed 11), it judged 2 as
  real asks (1.3%, Wilson 95% 0.4-4.7%) and 7 as borderline; none of the 2 is a clear AI ask.
  Files: neg_sample.jsonl, neg_judgements.jsonl.
- Side finding: one person with a LessWrong and an EA Forum account counted as two askers. Fixed as
  DEVIATIONS D6.
- Not done: the scan of texts the regex skipped (scan_regex_misses.py was written but did not finish).

No sign-off.
