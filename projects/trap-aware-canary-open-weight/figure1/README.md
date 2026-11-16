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
