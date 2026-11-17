# Figure 1 (Owain), prebuilt for the continuation room

Prebuilt in room theory-1117-head-clamp-8b while the 8B jobs waited for the laptop queue. Every number comes from
the saved files; `gen.py` asserts each one. The design follows the published 1.7B Figure 1
(blog.kran.ai/head-clamp-retrain): one real test prompt, the robot's answer with and without the probe, one row per
clamped head set.

## Files

- `pick.py`: picks the example test prompt by the rule posted before any 8B row was read (room #23537). Prints the count of prompts that qualify.
- `gen.py`: writes `figure1.html` from `$DATA/runs/*.json`, `$DATA/audit/s*.json`, `data/band17b/runs/*.json` (1.7B at the 64-update grid) and the published 1.7B `status.json` (pinned commit, `git show`).
- `text.json`: all words. Set `example` (from pick.py), `head`, `kicker` and `layout` here.
- `render.sh`: gen.py + the blog's own renderer (`_thumbs/figure1.mjs`) into `assets/figures/head-clamp-retrain-8b-fig1/`.

## Three modes (gen.py chooses by the files that exist)

1. **Retrain cells exist** (`data/runs/set_s0.json`): trained-switch row, then one row per head set (audit set, all but one, all 64). Each row has the 8B verdict, the 64 clamped-head dots and an 'At 1.7B' line.
   - `layout: v1`: answer columns = 8B right after clamping | 8B after training again (as the 1.7B figure).
   - `layout: v2`: answer columns = 1.7B after training again | 8B after training again, same prompt, same arm and seed (the 1.7B-vs-8B contrast in the example itself). The note says both answer B right after clamping (asserted).
2. **Formation null** (switch runs, no retrain cell): two rows, Qwen3-1.7B learned the switch vs Qwen3-8B did not, same prompt.
3. Rows whose files are missing are left out; counts say 'so far' when fewer than 3 models ran.

## Steps for the continuation room

1. `cd kran-research && git pull --rebase --autostash` (the laptop worker commits `data/` and `adapters/`).
2. `cd projects/head-clamp-retrain-8b/figure1 && LAYOUT=v2 python3 pick.py 0`: put the top prompt id in `text.json` `example` (if 0 qualify, try seed 1 and set `seed`).
3. Write `head` (how the result looks in the example, plain words, at most one number, not the post title) and `kicker` ('Question: ... Answer: No. / Yes. / Not settled, because ...'). Head drafts below.
4. `BLOG=<blog clone> ./render.sh`; open figure1.png and og.png; run the oracle Figure 1 gate (`~/bin/oracle.py check`, or readq.py from the room's swarm dir) against the filled TLDR.
5. Commit `assets/figures/head-clamp-retrain-8b-fig1/{figure1.html,figure1.png,og.png}` and `_data/figure1.json` with the post; front matter `figure1: head-clamp-retrain-8b-fig1` and `figure1_alt`.

## Head drafts by outcome (check each against the data before use)

- Audit set came back, all 64 held: 'After training again, Qwen3-8B answers A with the audit's heads clamped. Only clamping all 64 heads keeps it at B.'
- Audit set held: 'After training again with the audit's heads clamped, Qwen3-8B still answers B.'
- Formation null: 'The training that taught Qwen3-1.7B the switch did not teach it to Qwen3-8B.'
