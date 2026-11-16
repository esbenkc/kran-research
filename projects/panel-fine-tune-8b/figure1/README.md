# Figure 1 for panel-fine-tune-8b (Owain, room oracle-1116-panel-fine-tune-8b)

Status: prebuilt and tested on the 1.5B data (layout) and on the Qwen3-0.6B smoke (file format).
The real figure waits for the 8B laptop jobs (results.json with the 8B arms).

## Files
- `pick.py`: picks the example from saved scores. The example comes from the COMMONEST pattern
  among false statements that the original panel catches (C|mmm = caught by the original, missed by all 3
  fine-tunes; C|CCC = still caught by all 3). It needs the right country first in every state. It writes
  per-readout z scores for the false statement AND its true twin, plus each state's panel alarm line.
- `gen.py text_vNN.json`: draws `blog.kran.ai/assets/figures/panel-fine-tune-8b-fig1/figure1.html`.
  It asserts the 1.5B numbers (99%, 3.7%, 64%) from results.json at edb6cfa2b4 on
  swarm/theory-1022-panel-max-attack, and asserts the example's pattern equals the modal one.
- `text_test15.json`: the layout test on 1.5B data (NOT for the post).
- `gatepng.py`: the oracle FIG1 gate on a PNG against `_notes/panel-fine-tune-8b.md` (with-model-keys).

## Steps once the 8B jobs are in
1. `git pull --rebase --autostash` in kran-research (the labq worker commits the outputs).
2. From `kran-research/projects/panel-fine-tune-8b`:
   `uv run --no-project --with numpy --with pyyaml --with transformers --with torch python <fig1>/pick.py --main panel__lam10000__seed0 panel__lam10000__seed1 panel__lam10000__seed2`
   then pick a short, familiar city from the printed list and rerun with `--pick IDX --out <fig1>/fig1_data.json`.
3. Copy `text_test15.json` to `text_v01.json`: project = the 8B project, example = fig1_data.json,
   model_label "Qwen3-8B", last_layer 36, kept_note with the twin's real country, table rows
   (1.5B ref / 8B random-direction control / 8B truth panel as the key row), `example_caught_after`
   true or false, head and subtitle from the TLDR (head = how the example looks; subtitle = the post's
   question in a few words + "Answer: ..." + the key numbers).
4. `python3 gen.py text_v01.json`, then from the blog clone root
   `CHROME_PATH=... node _thumbs/figure1.mjs panel-fine-tune-8b-fig1`; check figure1.png and og.png.
5. Gate: `with-model-keys python3 gatepng.py <png> 6 <blog clone>/_notes/panel-fine-tune-8b.md`; then room done with Selma and Lumi as reviewers.

## Review notes already in (room #22304, #22305, #22334)
- Show both lines in the fine-tuned card (original, dashed; the model's own, red). Done.
- The blue true bars must visibly reach the dashed original line when they do (Selma #22334).
- Table: the fix column says what the fix is ("fix: scale set again"); include the random-direction control row.
- 8B axis reads 13..36 (readouts come from common.PANEL / results.json).
- Never write "blind" without "at the model's own line" (Tove #22310).
