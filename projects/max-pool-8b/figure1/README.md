# Figure 1 for max-pool-8b (Owain, room oracle-1119-max-pool-8b)

One real city pair, the sentence as the x axis. Dots = the clean-fit max-pooled probe's score on each token
(`scores.npz:tok_max`), drawn against the state's own alarm line (2nd-highest max score of the 200 refit true
statements, `common.threshold`). Left card: the original model. Right card: main seed 0 at its gate-scored step.
The highest token gets a pill (red when it crosses the line); a bracket marks the shared words ("same score in the
true sentence"). Big numbers: clean max18 catch and the pooled catch of the 3 main seeds, each seed listed.
Notes: the 1.5B parent (91% -> 0%, results.json on swarm/theory-1019-max-pool-survives) and the random-direction
control (max18 per seed).

## Pick rule (frozen in the room, #26136, before any 8B data)
Among eval 'cities' FALSE statements that the clean max probe catches at the clean line, take the commonest
caught/missed pattern over the 3 main seeds; inside it, sort by distance of the seed-0 statement score to the group
median; choose a familiar city only among the first 40 of that list (a single-token city name when one is there).

## Files
- `pick8.py`: prints the patterns and the candidate list; `--pick IDX --tag 8b` writes `ex_8b.json` (tokens and
  per-token scores of the example and its true twin, every main state, each line) and `numbers_8b.json` (catch
  rates; asserts they equal `data/results/results.json` pooled.main.max18, per_state.clean.max18 and
  pooled.random.max18; asserts the parent 91% -> 0%).
- `gen.py text_vNN.json`: draws `<blog>/assets/figures/max-pool-8b-fig1/figure1.html`. Asserts the shared-prefix
  fact on the example (shared tokens score the same in the twin), that the example is caught by the clean probe,
  and that the wording's claims (caught after or not; highest token after shared or not) match the data.
- `text_smoke.json` / `ex_smoke.json` / `numbers_smoke.json`: the pipeline test on Bruno's 0.6B smoke (NOT for the post).
- `text_test15.json` / `pick15.py`: the layout test on the 1.5B parent data (NOT for the post).

## Steps once the laptop jobs are in
1. `git pull --rebase --autostash` in kran-research (the labq worker commits the outputs).
2. From `projects/max-pool-8b`: `uv run --frozen python figure1/pick8.py`, choose IDX from the printed list, then
   `uv run --frozen python figure1/pick8.py --pick IDX --tag 8b`.
3. Copy `text_smoke.json` to `text_v01.json`: example `ex_8b.json`, numbers `numbers_8b.json`, fig_id
   `max-pool-8b-fig1`, model_label `Qwen3-8B`, seeds1 `[main_s0, main_s1, main_s2]`, `example_caught_after` and
   `top_after_shared` from the data, the kicker answer from the post's TLDR, the head from how the example looks.
4. `python3 figure1/gen.py figure1/text_v01.json`, then from the blog clone root
   `CHROME_PATH=... timeout 150 node _thumbs/figure1.mjs max-pool-8b-fig1`; check figure1.png and og.png.
5. Gate: `with-model-keys python3 figure1/gatepng.py <png> 6 <blog clone>/_notes/max-pool-8b.md`; cold read with
   `coldread.py`; then room done with Selma and Lumi as reviewers.
