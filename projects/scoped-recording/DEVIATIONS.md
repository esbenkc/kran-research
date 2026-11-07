# Deviations — scoped-recording

Every change to a rule frozen in PREREG.md (v1 5ede5d20, v1.0.2 35f48f86). The frozen rule stays a reported row.

## D1 — manipulation-check reply format (before seed 100; Felix, from Bruno #16046)

**What changed.** M1 and M2 (PREREG s5) keep their frozen question text byte for byte. One line is added after
each: `Reply with that one word only. Do not write JSON and do not call a tool.` The parser does not change.

**Why.** The replay call keeps the arm's system prompt, which tells the agent to return one JSON object with
thought/tool/args. With the frozen text alone, 41 of 60 smoke replies could not be parsed: agents put the answer
inside thought or args, or kept doing the task. Scored literally, P5 would fail on reply format, not on what the
agent read. With the added line (Bruno, `scratch/manip_format.py`, smoke seeds only) 0 of 60 replies were
unparsed. The decision uses the parse count only. Disclosure: Bruno's first read of smoke answers (Haiku answered
'both' to M1 in the key and flag arms) was posted in the same message; the format line does not target answer
content.

**Frozen rule kept as a reported row.** The smoke-only parse rate of the frozen text (41/60 unparsed, per agent and
arm in `results_smoke/manip_summary.json`) is reported. The frozen text is not re-run on counted seeds (cost rule
s7). P5 is scored on the D1 version; the post says so where it reports the prediction outcome.
