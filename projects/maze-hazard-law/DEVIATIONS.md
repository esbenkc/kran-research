# Deviations from PREREG.md — maze-hazard-law

Every change or clarification after the prereg push (f128b078c, GitHub push 2026-11-18T23:08:27Z). A line that starts
with "RULE CHANGE:" changes the FROZEN RULE block and makes the run EXPLORATORY.

- **D1 (2026-11-19 00:25 CET, Iris; before any model call; reported row only, outside the FROZEN RULE).** Spend row
  (s.4 reported rows): S_r counts calls from the entry call on, and the entry call has depth 0 (no page read before it).
  The spend line is therefore fitted on all pilot pairs with j <= 3 including j = 0, and the predicted median spend sums
  j = 0..ceil(M_pred) (PREREG wrote j = 1..ceil(M_pred), which would leave out the entry call that S_r includes).
- **D2 (2026-11-19 00:30 CET, Iris on Kasper #25830; before any maze or dead-end run: data/ held only Qwen3-1.7B
  visible-arm gate files).** Clarification of an undefined case in the FROZEN RULE, not a change of a threshold: the
  shape check passes only when both tests return a p-value. If one test has no p (all decisions fall in one page bin,
  or every decision is a give-up) and the other does not FAIL, the law is UNTESTED for that agent (named outcome
  "shape untestable"), never HOLDS. Reason: an agent that always leaves on page 1 would otherwise "hold" a constant
  hazard that was never observed past page 1. New reported rows: the share of give-ups on pages 1-3, and a flag when
  the Kaplan-Meier median is <= 3 (the pilot then contains the median and the 2x check is circular; the post says so).
- **D3 (2026-11-19 00:20 CET, Bruno, decided by Felix #25843 with Iris #25838 and Vera #25841; before any maze or
  dead-end run: data/ held only visible-arm gate files of qwen3-1.7b-nothink, gemma-3-1b and llama-3.2-1b; outside
  the FROZEN RULE).** PREREG s.1 allows prefix KV reuse "only after a smoke shows bit-equal outputs with and without
  reuse". The smoke (`web/test_prefix.py llama-3.2-1b 20 greedy`, toy transcript with no maze page, commit d572ad634)
  gave 18 of 20 replies byte-identical with reuse ON vs OFF; the 2 that differ are calls 17 and 18, both >1,000-token
  rambling greedy decodes, where a different floating-point summation order (incremental vs full prefill) flips a
  near-tie. Reuse after call 0 fed at most 305 new tokens per call. So reuse is numerically equal, not bitwise equal.
  Reuse is kept: without it a capped 80-call run re-reads ~1.1M prompt tokens (`web/measure_context.py`). At
  temperature > 0 the sampled replies come from the same distribution up to that floating-point noise. The cache is
  reset at each run's first call, so a run never depends on the run before it. Follow-up evidence (logit difference at
  the first differing token, and a no-reuse vs no-reuse control) is added below when measured.
  D3 follow-up (00:24 CET, `web/test_prefix_logits.py <model> <call>`: last-position logits of call t with the
  trimmed call t-1 cache plus the new tokens, vs a fresh full prefill, and a second full prefill as the same-host
  control): llama-3.2-1b call 5 (1,727 tokens, 305 fed) max |diff| 0.027, call 17 (5,291 tokens) 0.048, on logits up to
  ~17; full vs full 0.0 both times; argmax equal; the top-2 gap at call 17 is 0.156, the size of a near-tie that 0.05
  can flip in a long decode. qwen3-1.7b-nothink call 17 (5,377 tokens, 321 fed): max |diff| 0.0, bit-equal. So the
  2/20 is reuse-order fp noise in Llama's bf16 path, not nondeterminism of MLX itself (full prefill repeats exactly).
- **D4 (2026-11-19 00:32 CET, Bruno, decided by Felix PLAN v1.1 #25862 with Kasper #25857, Tove #25866, Vera #25863;
  before any maze or dead-end run of any agent; outside the FROZEN RULE).** All three frozen mini side agents failed the
  frozen visible-link gate (>= 8/10): qwen3-1.7b-nothink 1/10 (data/qwen3-1.7b-nothink/job.log, 23:14:45Z); llama-3.2-1b
  0/1 when stopped (run 000 used all 80 calls and never answered, 23:18:00Z); gemma-3-1b 0 runs finished (each call
  echoed ~2,300 tokens of prompt JSON, 45 s per call; stopped). A thinking-on Qwen3-1.7B (qwen3-1.7b-think, not in
  PREREG s.1) was started as a last mini try and stopped by the planner after 3 gate runs, 0/3 correct; its gate files
  stay on the branch, and it has no maze run. Added: ONE closed-model side row, gpt-5-nano through the OpenAI API
  (`web/openaicall.py`; reasoning_effort minimal, API-default sampling, prompt_cache_key = run id), the same harness,
  prompt, cap 80, arms and frozen rule; frozen gate >= 8/10 first, and if it fails no other model is tried. It is NEVER
  a status agent and is not open-weight: it is a pipeline check of the frozen rule and analysis.py on real agent
  transcripts. Design: maze 40 (3 seed blocks) + dead end 20 (2 blocks), one process, 4 runs in flight
  (`run.py --threads 4`), no new run after 02:40 CET. Spend cap USD 2.00 (inside the 5.00 experiment budget), checked
  before each run start from the billed usage of every cached call (`openaicall.spent`); every call is cached.
