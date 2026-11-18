# Figure 1: trap-aware-canary-open-weight

Builds `blog.kran.ai/assets/figures/trap-aware-canary-open-weight-fig1/figure1.html` from this project's raw runs.
Every quote, footer line, log line and number is read from files here and checked by `gen.py` (asserts).

Tonight's version (2026-11-16, room defense-1116-trap-aware-canary) is a placeholder: its example is Qwen3-1.7B,
a side model that failed the competence gate (it opened a shown link in 0 runs). The two models the question names
were still in the laptop queue. **The continuation must swap the example to gpt-oss-20b before the post ships.**

## Files

- `gen.py`: writes figure1.html. Reads `web/out/<agent>/{raw,access.log,scores.jsonl}`, `results/parent_api_cells.csv`,
  and cross-checks `results/cells.csv` when present.
- `pick.py`: prints the outcome-pattern counts over (no warning/hidden, warning/hidden, warning/shown) at equal run
  index, and the candidate runs with their reasoning lines that match the frozen HIDDEN_RE.
- `text.json`: every word in the figure. `quotes[cell]` must be verbatim substrings of that run's reasoning or reply
  text, up to the step that fetched the hidden link (gen.py asserts this).
- `render.sh`: gen.py + the site renderer (`_thumbs/figure1.mjs`). Restores `_data/figure1.json` unless `FINAL=1`.

## Swap to gpt-oss-20b (about 10 minutes once `web/out/gpt-oss-20b/scores.jsonl` exists)

1. Pick the example by the rule fixed before any trace was read (room #22486): the agent's MODAL pattern, then the
   LOWEST run index with that pattern and all 3 cells complete.
   `uv run --frozen python figure1/pick.py --agent gpt-oss-20b --want neutral-cued:X,warning-cued:Y,warning-none:Z`
   (first run it with any --want to see the pattern counts; set X,Y,Z to the modal pattern).
2. In `text.json`: set `quotes` for the 3 cells of that run (short verbatim pieces: the sentence where it names the
   link or its hiding, and the sentence where it decides), `head` (how the result looks in that run, plain words,
   at most one number, not the post question or title), `sub` (the post question + "Answer: Yes./No./Not settled."
   + the two agents' warned and unwarned rates with the gpt_oss_20b_* and qwen3_8b_* keys), `start`/`slot` stay.
   `dotnote_qwen3-1.7b` stays ("could not do the task").
3. `cd projects/trap-aware-canary-open-weight && uv run --frozen bash figure1/render.sh v10 --agent gpt-oss-20b --i <i>`
   and open `/tmp/v10.png` and `/tmp/v10-og.png`. If the answer for a card is not "says the page has no price",
   set `act_yes_cued` / `act_no_cued` / `act_yes_none` / `act_no_none` to match what the run did.
4. Cold read: give only the PNG to a cheap model (`with-model-keys python3 figure1/coldread.py /tmp/v10.png /tmp/cold.json`
   from the swarm owain dir, or `oracle.py check`) and confirm it states the question and the answer.
   Re-gate after every TLDR edit: a passing figure fails when the TLDR numbers change.
5. Final: `FINAL=1 ... render.sh v11 ...`, then commit figure1.html, figure1.png, og.png and `_data/figure1.json` on the
   blog branch; set `figure1_alt` in the post front matter to a plain description of the example and the result.

Sources of tonight's placeholder: `web/out/qwen3-1.7b/raw/{neutral-cued,warning-cued,warning-none}/000.json`,
`web/out/qwen3-1.7b/access.log`, `web/out/qwen3-1.7b/scores.jsonl`, `results/parent_api_cells.csv`
(parent branch swarm/defense-1110-trap-aware-canary at af50d21f).

## Room defense-1117-trap-aware-canary (2026-11-17)

- Example agent: gpt-oss-20b-d4 (the D4 rerun, labq c76l6dj5vo) by the same pick rule (room #23256 = #22486).
- Fallback tie-break, fixed 2026-11-16 23:05 PST before any further Qwen3-8B trace was read (Kasper #23329 (b)):
  if Qwen3-8B must be the example, skip a run where any panel's behaviour (opened no page at all) occurs in under
  5% of its cell, then take the next lowest i with the modal pattern. Qwen3-8B i=3 (rule pick) fails it: its
  warned-hidden run opened no page, 2 of 60 warned-hidden runs do that.
- gen.py (this room): every card prints the run's real first request of its hidden-link code (any status; a 404 is
  drawn as a request, never as 'did not follow'); chart rows `agent:tok` give the token reading (any GET whose path
  holds the run's link code, DEVIATIONS R2, defined after the Qwen3-8B data); both rows are cross-checked against
  results/cells.csv (events fit and fit_tok).

## Room defense-1118-trap-aware-canary (2026-11-18): shipped on gpt-oss-20b

- Example: gpt-oss-20b-d4 run i=0 (Northwind Cloud), by the pick rule posted before any trace (room #24075): modal
  pattern (1,1,1) over (neutral-cued, warning-cued, warning-none) on the frozen exact metric, 54 of 60; lowest i.
- Text: `text_gptoss.json`. Render:
  `python3 figure1/gen.py --agent gpt-oss-20b-d4 --i 0 --text figure1/text_gptoss.json --rows parent,gpt-oss-20b-d4,gpt-oss-20b-d4:vis,qwen3-8b`
  then, from the blog clone, `timeout 150 node _thumbs/figure1.mjs trap-aware-canary-open-weight-fig1` (the renderer
  can hang after writing the PNGs on a busy machine; the files and the _data entry are already written).
- gen.py now also reads every event of results/cells.csv as `{cc_<agent>_<cell>_<event>_k}` (e.g. knowing_rt, fit_correct),
  cross-checks knowing_rt against scores.jsonl, asserts 'never mentions traps' over every step (TRAP_RE) and the right
  answer for any card that prints it.
- The Qwen3-8B any-address row was dropped (room #24403, #24407): without its control pages it reads as a warning effect.
