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

## Shipped (room oracle-1120-max-pool-8b-cont)
- Example: Kyoto (idx 1065), rank 26 of 40 in the 3-seed modal pattern C|mmm (289 of 298 eval cities); single-token
  city, same pair as the post text. In all 3 main seeds every token of the false sentence is below that seed's line and
  the highest token is the shared ' Kyoto' (pick8.py prints each seed's per-token scores).
- `uv run --frozen python figure1/pick8.py --pick 1065 --tag 8b` (from projects/max-pool-8b) wrote `ex_8b.json` and
  `numbers_8b.json`; it asserts them equal to data/results/results.json at 60992fa5e (status REJECTED).
- `python3 figure1/gen.py figure1/text_rej.json` (REJECTED ending, shipped) or `text_unc.json` (the 2-seed 'Not
  settled' ending, kept for the record); gen.py refuses to render while a text json holds `[FILL`.
- Render from the blog clone: `timeout 150 node _thumbs/figure1.mjs max-pool-8b-fig1`.
- Gate: `with-model-keys python3 figure1/gatepng.py <png> 6 <blog>/_notes/max-pool-8b.md` -> 6/6 PASS on the post at
  blog 5e1f2381; PNG-only cold read (2 cheap models x 3): `cold_rej.json`, 6/6 state the question and 'No'.
- Removed: the 1.5B layout test (text_test15/ex15/numbers15/pick15), superseded by the real 8B figure.
