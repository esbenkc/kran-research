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
- **D3 (2026-11-19 00:17 CET, Bruno, decided by Felix #25843 with Iris #25838 and Vera #25841; before any maze or
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
  D3 follow-up (00:20 CET, `web/test_prefix_logits.py <model> <call>`: last-position logits of call t with the
  trimmed call t-1 cache plus the new tokens, vs a fresh full prefill, and a second full prefill as the same-host
  control): llama-3.2-1b call 5 (1,727 tokens, 305 fed) max |diff| 0.027, call 17 (5,291 tokens) 0.048, on logits up to
  ~17; full vs full 0.0 both times; argmax equal; the top-2 gap at call 17 is 0.156, the size of a near-tie that 0.05
  can flip in a long decode. qwen3-1.7b-nothink call 17 (5,377 tokens, 321 fed): max |diff| 0.0, bit-equal. So the
  2/20 is reuse-order fp noise in Llama's bf16 path, not nondeterminism of MLX itself (full prefill repeats exactly).
- **D4 (2026-11-19 00:22 CET, Bruno, decided by Felix PLAN v1.1 #25862 with Kasper #25857, Tove #25866, Vera #25863;
  before any maze or dead-end run of any agent; outside the FROZEN RULE).** No frozen mini side agent passed the frozen
  visible-link gate (>= 8/10); only one completed it: qwen3-1.7b-nothink 1/10 (data/qwen3-1.7b-nothink/job.log, 23:14:45Z); llama-3.2-1b
  stopped for speed at 0/1 (run 000 used all 80 calls and never answered, 23:18:00Z; 8/10 was still reachable); gemma-3-1b stopped for speed with 0 runs finished (each call
  echoed ~2,300 tokens of prompt JSON, 45 s per call; stopped). A thinking-on Qwen3-1.7B (qwen3-1.7b-think, not in
  PREREG s.1) was started as a last mini try and stopped by the planner after 3 gate runs, 0/3 correct; its gate files
  stay on the branch, and it has no maze run. Added: ONE closed-model side row, gpt-5-nano through the OpenAI API
  (`web/openaicall.py`; reasoning_effort minimal, API-default sampling, prompt_cache_key = run id), the same harness,
  prompt, cap 80, arms and frozen rule; frozen gate >= 8/10 first, and if it fails no other model is tried. It is NEVER
  a status agent and is not open-weight: it is a pipeline check of the frozen rule and analysis.py on real agent
  transcripts. Design: maze 40 (3 seed blocks) + dead end 20 (2 blocks), one process, 4 runs in flight
  (`run.py --threads 4`), no new run after 02:40 CET. Spend cap USD 2.00 (inside the 5.00 experiment budget), checked
  before each run start from the billed usage of every cached call (`openaicall.spent`); every call is cached.
- **D5 (2026-11-19 00:31 CET, Bruno on Vera #25927; no status-agent run exists; outside the FROZEN RULE).** The
  laptop chunks n8zed9xgcv and bge9lj084j (commit 09f25132c) were cancelled while still queued and resubmitted as
  f0jxnah2hu (gpt-oss-20b-low) and erbnwpu5t3 (qwen3-8b-nothink) on commit 8698d55b2, last in the queue as before (no
  job was behind them). The only change for these agents: `run_laptop.sh` first runs `web/test_prefix.py <agent> 12
  greedy-short` (toy transcript, greedy, short replies) and writes `data/<agent>/prefix_check.json`, so the prefix
  reuse path of gpt-oss-20b (sliding-window cache snapshots, never run on gpt-oss before) is checked on the machine
  that runs it. If that check fails (fewer than 95% identical replies), the continuation reruns that agent with
  `MAZE_PREFIX_REUSE=0` and logs it here; analysis.py itself is unchanged. On the mini, the same check on Gemma-3-1B's
  sliding-window cache gave 6 of 6 and 8 of 8 identical replies.
- **D6 (2026-11-19 07:53 CET, Bruno; continuation room defense-1119-maze-hazard-law; outside the FROZEN RULE; applies
  the D5 rule as written).** Both laptop chunks of the parent room ran after its close (commits 8c8761af2, fc6b15c97).
  (1) Both FAILED the D5 prefix-reuse check on the laptop: gpt-oss-20b-low 5 of 12 identical greedy replies
  (`data/gpt-oss-20b-low/prefix_check.json`; the stored 160-character heads of the differing replies are identical,
  so they part later in the decode), qwen3-8b-nothink 9 of 12 (`data/qwen3-8b-nothink/prefix_check.json`); the bar
  was 95%. As D5 fixed before any status-agent run, both agents are rerun with `MAZE_PREFIX_REUSE=0` (full prefill on
  every call) into `data/<agent>-full/` (`bash run_laptop.sh <agent> <h> <maze n> <deadend n> 0`). `analysis.py`
  counts a reuse-on folder whose `prefix_check.json` says `pass: false` only as a labelled side row ("prefix reuse
  on, failed D5 check; not counted"), never as a status record. The reuse-on records stay on the branch.
  (2) gpt-oss-20b-low's chunk crashed after the gate (9/10) and 9 maze + 8 dead-end runs: the model wrote a fetch URL
  with a non-ASCII character ('…'), and Python's http.client refused to send it (UnicodeEncodeError). Fix in
  `web/agent.py` `_http_get`: characters outside printable ASCII (and space) are percent-encoded before the request;
  the sandbox server answers 404, as for any other unknown page. A printable-ASCII URL is sent unchanged, so no earlier
  record would differ. Checked with the sandbox server: '/m/…' -> 404, '/m/a b.html' -> 404, '/' -> 200;
  `web/test_harness.py` all checks OK.
- **D7 (2026-11-19 08:05 CET, Bruno; before any full-prefill run; outside the FROZEN RULE; target n only).** Qwen3-8B
  dead-end target 20 -> 12 (the floor is 2 blocks x 6) for the full-prefill rerun. Reason, from the reuse-on records:
  9 of 20 Qwen dead-end runs hit the 80-call cap, and a capped run is ~1.07M prompt tokens (data/qwen3-8b-nothink/raw/
  deadend/009.json: 1,097,712). With full prefill at ~400 prompt tokens/s on the laptop (prefix_check.json: 106 s for
  12 calls of up to 3.5k tokens), one capped run takes ~45 min. With 20 dead-end runs interleaved by run index, the clock
  would stop the job before the maze floor (3 blocks x 10 entered). Maze target stays 40; gpt-oss-20b targets stay
  60 + 20 (its reuse-on dead-end runs used 4-10 calls). Job: labq oka1mlfbod (1.5 h, 40 + 20) cancelled while queued;
  resubmitted as labq 2vqcuc4w2n, `bash run_laptop.sh qwen3-8b-nothink 6 40 12 0`, estimate 5.5 h. gpt-oss-20b-low full: labq dx0yscom1o (`bash run_laptop.sh gpt-oss-20b-low 2.5 60 20 0`, estimate 2 h).
  D7 addendum (08:15 CET): dx0yscom1o was submitted with `--model openai/gpt-oss-20b`, but the harness loads
  `mlx-community/gpt-oss-20b-MXFP4-Q8` (web/mlxcall.py:59; the worker sizes disk and downloads from that field). Both
  jobs were cancelled while queued (no job behind them) and resubmitted on commit bd997c32a, same commands:
  56pkb4cysk (gpt-oss-20b-low, mlx-community/gpt-oss-20b-MXFP4-Q8, 16 GB, 2 h) and fdce5qspgd (qwen3-8b-nothink,
  Qwen/Qwen3-8B, 11 GB, 5.5 h).
- **D8 (2026-11-20 08:02 CET, Iris; continuation room defense-1120-maze-hazard-law; decided by Felix FINAL PLAN v1
  #26814 and #26820 with Tove #26811/#26816, Kasper #26808, Vera #26822; written and pushed BEFORE the job below is
  submitted and before any no-page-number run exists; outside the FROZEN RULE; a reported row, never a status input).**
  The status is already set by the frozen rule on existing data: REJECTED via gpt-oss-20b (shape check fails: Pearson
  p 9.6e-12, discrete Weibull k 3.07, profile 95% [2.48, 3.72], LR p 1.6e-16; the 2x miss R 6.5 is not decisive).
  Nothing in this entry can change that status, the prediction score, or Figure 1's headline.
  *Why.* Every counted maze page prints its page number ("part k" in the title tag, the heading and the link anchor).
  In 17 of 53 give-up calls the model's own reasoning names a part number, and 8 of 53 cite "time" with no clock or
  step counter in the prompt (`research_iris/quit_cues.py`, `results/quit_cues_gpt-oss-20b-low-full.json`). So the
  rising hazard could be a reaction to the printed count. One arm that removes only the count tests that.
  *Arm `maze_nolabel`* (`web/sites.py`, Bruno, commits 14de385b7 and 01f062093 (`web/sites.py:210-221`, `web/run.py`, `analysis.py:487`, `nolabel_analysis.py:38-49`, `run_nolabel.sh`)): the counted maze page with the number k removed
  everywhere: title "<route> 2025, continued", heading "<fit anchor>: continued", link "<fit anchor> (next part)". The
  body sentence and the promise "Figures are published in a later part." stay word for word; the rest of the site is
  byte-identical (render diff: `results/d8/render_diff.txt`). Same agent (gpt-oss-20b-low, mlx-community/
  gpt-oss-20b-MXFP4-Q8, reasoning effort low, temperature 1.0), same prompt, tools, 80-call cap, full prefill
  (MAZE_PREFIX_REUSE=0), same seed and block formula as the counted maze arm (run i has the same template, task wording,
  seed block and call seeds as counted maze run i). 39 runs (3 seed blocks x 13; Bruno #26827: at the 90% entry rate 13 runs give ~11.7 entered per block, above the floor of 10), maze arm only, no dead-end runs.
  The visible-link gate is not rerun: the gate of the same model and settings (`data/gpt-oss-20b-low-full`, 10/10) is
  read with `run.py --gate-dir`. Outputs: `data/gpt-oss-20b-low-nolabel/` (arm `maze_nolabel`). `analysis.py` skips
  folders ending in `-nolabel` (one-line change, so the status inputs are unchanged); the row is scored by
  `nolabel_analysis.py` with analysis.py's own `load_agent`, `law` and `risk_tables`. Job (cwd `projects/maze-hazard-law`): `bash run_nolabel.sh 1.25 39` (awake budget 1.25 h, then it starts no new run), laptop queue, `--mem-gb 16` (13.13 GB peak for the same model and settings, `data/gpt-oss-20b-low-full/job.log`), estimate 1.0 h; compute gate PASS Vera #26833.
  *Reading rule (frozen now).* On the no-page-number arm's entered runs, with L_lo = 2.48 (the lower end of the
  counted arm's pooled profile 95% interval for k):
  - "no rise detected" (the label-driven reading) iff the arm's k profile 95% interval includes 1 AND its upper end is
    below 2.48. Plain words in the post: "without page numbers we saw no rise; a weak rise (k near 1.5) is not ruled
    out" (Vera `skeptic/vera_d8_sim.py`: a true k = 1.5 reads this way 14-15% of the time, 38% with long stays);
  - "rise survives" iff the interval's lower end is above 1;
  - "falling" iff the interval's upper end is below 1;
  - "unclear" otherwise, and also when the arm has fewer than 10 give-ups or no Weibull fit.
  Simulated reads at 36 entered runs (close to the ~35 expected) (Vera #26822, 150 sims per world): true k = 1 -> "no rise" 0.91-0.95; true
  k = 2 or 3.07 -> "rise survives" 1.00; true k = 0.6 -> "falling" 0.95. Reported next to the reading: entered runs,
  give-ups, share at the 80-call cap, Kaplan-Meier median pages (vs 12 with numbers), h3, h_all, k with its interval,
  Pearson p, the per-page table, and the paired exit-page difference by run index.
  *Limit, frozen now.* Transcript length grows with depth in both arms, so "rise survives" cannot separate depth from
  transcript length; only "no rise detected" is informative about the label (Kasper #26808 (3)).
  *Guard.* If the job has not started by 10:00 CET (01:00 PST) or is not done by 11:30 CET (02:30 PST), it is
  cancelled (`labq cancel`); then the post says nothing about this arm, any partial runs stay on the branch unscored,
  and the close stays REJECTED. If it lands, the post gives it ONE sentence and one proof-matrix row.
  D8 addendum (2026-11-20 08:05 CET, Iris on Kasper #26852; job 7mvin1ebxm still queued, no no-page-number run
  exists; wording of the reading only, the rule and its code are unchanged): the "Limit" sentence above undersells
  "rise survives". Read the outcomes as: "rise survives" = the page number is not needed for the rise; it cannot tell
  depth from transcript length. "no rise detected" = the page number drives most of the rise (a weak rise, k near 1.5,
  is not ruled out).
  D8 guard change (2026-11-20 08:16 CET = 23:16 PST, Iris on Felix #26924 with Tove #26921, Kasper #26922, Vera
  #26923; before any no-page-number run: `labq list` at 07:15Z shows 7mvin1ebxm queued, worker esben-laptop last seen
  07:00:10Z after reporting about every 7 min, so the laptop is likely asleep; timing of a reported row only, the
  reading rule and its code are unchanged): the job is cancelled if not STARTED by 09:30 CET (00:30 PST), and if
  started, cancelled if not done by 11:15 CET (02:15 PST). The post and the close do not wait for it: the post ships
  without the D8 sentence. If the job lands in time, it enters as one follow-up PR (one sentence and one proof-matrix
  row, after Vera rebuilds the row).
  D8 job record (2026-11-20 09:15 CET, Iris on Bruno #26987 and Vera #26986): 7mvin1ebxm started 07:39Z and FAILED
  at 08:10:50Z ("no heartbeat for 10 min, 3 attempts"; the laptop slept). Its log ends at "GATE: 10/10 correct:
  passed"; no maze_nolabel run exists and nothing was pushed. Bruno resubmitted it once as fbqxvuw7wi (created
  08:11:13Z, commit b9165dd7e, same command `bash run_nolabel.sh 1.25 39`, same code and D8 rule), under the same
  guard (cancel if not started by 00:30 PST or not done by 02:15 PST).
