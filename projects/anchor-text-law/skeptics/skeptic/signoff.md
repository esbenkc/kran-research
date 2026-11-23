role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: defense-1123-anchor-text-law

**Answer to the question:** NOT SETTLED. With gpt-oss-20b and Qwen3-8B, a law fitted on 4 of 8 cells (follow from
the model's anchor-carrying fit cells x reach from the other model's fit cell) predicts 2 of the 3 anchor-carrying
held-out cells within 10 points. It misses Qwen3-8B's raw-HTML cell by 23 points, and that miss is not wholly
outside the 10-point band by the frozen interval test (hi -0.094 vs the -0.10 line). **SIGN-OFF: UNCONFIRMED (the
frozen status reproduces).** Pre-registered prediction: failed (C1 false; C2 and C3 true).

The PASS above means: the build's numbers and status reproduce from raw data with my own code. It does not mean
the answer is yes.

## How I rebuilt it
My own code, from raw data only: `web/out/<slot>/<model>/raw/<cell>/NNN.json` (transcripts, site ids, answers) and
`web/out/<slot>/<model>/access.log`. I did not read or import analysis.py, run.py or any build script. Follow =
a logged GET whose path holds the run's canary id `hid` (any folder, any status), matched by id only, never by the
run-label column. Reach = `hid` in an `observation` message. Gate success = follow AND the final answer holds
`site.answer` (my own string check; it agrees with the harness `correct` field in 60/60 gate rows of the two scored models).
Scored set = largest index prefix complete in all 4 main cells, rounded down to a multiple of 3.

- `vera_check.py` (frozen rule s3-s5, exact Fractions) -> `out_check.txt`
- `vera_diff.py`: 134 fields vs `results/analysis.json` (cell counts, per-block counts, fr, catch, pred, err,
  seed-block errs, nulls, fr gaps, gates) -> **0 mismatches** (`out_diff.txt`)
- `vera_swap.py`: frozen B swap rebuilt: Qwen3-8B thinking off gate part 0 = 9/15 (< 11 -> swap fired, correct);
  thinking on 10/15; gpt-oss 15/15; no cap hits -> cap stays 16 (`out_swap.txt`)
- `vera_boot_edge.py`: stability of the knife-edge interval end (`out_boot_edge.txt`)
- `vera_null.py`, `vera_mech.py`: label-shuffle nulls and the mechanism of the Qwen raw-HTML misses
- `vera_power.py`, `vera_power2.py`: pre-data power (posted #28638, #28644 before the freeze)
- `vera_a_only.py`, `vera_mismatch.py`: J1 first look; found the 4 `/r/r/<hid>.html` follows (#28700)

## Key numbers (mine = build's)
| | gpt-oss-20b (A) | Qwen3-8B thinking on (B) |
|---|---|---|
| skill gate (visible link) | 30/30 | 22/30 (needs 21) |
| follow after reach: raw | 60/60 | 31/42 |
| markdown | 60/60 | 39/42 |
| bare list | 54/60 | 3/41 |
| anchor list | 60/60 | 41/41 |
| bare minus markdown (G needs <= -0.15) | -0.100 (no) | -0.855 (fires) |
| anchor minus markdown (needs within 0.10) | 0.000 | +0.071 (holds) |

Held-out status cells: A-markdown pred 1.000, err +0.000; A-anchor pred 41/42 = 0.976, err +0.024; **B-raw pred
1 x 80/83 = 0.964, catch 31/42 = 0.738, err -0.226, 95% bootstrap [-0.369, -0.094]** (mine, 2,000 draws, my own
RNG; build's frozen seed: [-0.3686, -0.0944]). Boundary cell B-bare err -0.892 [-0.964, -0.798].
Seed-level (t over 3 seed blocks, df 2): B-raw block errs -0.429 / -0.214 / -0.036, mean -0.226 [-0.715, +0.262].

## Checks that could have faked the result
1. **Knife edge.** REJECTED needs hi < -1/10 strictly. Over 500 bootstrap seeds, hi has median -0.0944 (range
   -0.106 to -0.083); P(hi < -0.10) = 0.016 (0.014-0.016 for three percentile conventions). A 200,000-draw bootstrap
   gives hi -0.0944. So UNCONFIRMED is the data's answer, not the frozen seed's luck. The post must not say "the
   law was rejected"; it may say "the law missed one cell by 23 points, just short of the pre-registered bar".
2. **Follow definition.** 4 gpt-oss bare runs (007, 017, 029, 054) requested `/r/r/<hid>.html` (404) after the
   off-task page; the harness flag `harness_followed` says no, the frozen event says yes. Frozen: 54/60, G does not
   fire for gpt-oss. With the harness flag it would be 50/60 and G would fire. The room uses the frozen count.
3. **Crash rerun (D1).** A bare 048's crashed attempt never requested its canary; crash-as-miss gives 53/60, gap
   -0.117: G still does not fire for A; status identical. No crash files in B.
4. **Label-shuffle nulls (`out_null.txt`, 20,000 shuffles).** Qwen bare-vs-markdown gap -0.855: never reached by
   shuffling (p < 1/20,000; G fires by luck in 9.1% of shuffles). gpt-oss gap -0.100: two-sided p 0.029.
   Anchor vs bare exchangeable for Qwen (`out_mech.txt`): the anchor clause passes in 0 of 20,000 shuffles. The
   seed-level t-interval excludes 0 in 5.3-5.6% of shuffles for Qwen (nominal 5%): it is calibrated here.
5. **Nulls vs the law (status cells).** Format-blind null err: A-md +0.050, A-anchor +0.073, B-raw -0.226; constant
   null: +0.049, +0.049, -0.213. In the two gpt-oss cells the nulls are also within 10 points: those cells sit at the
   ceiling (follow 60/60) and do not test the law. For Qwen the format-blind null equals the law by construction
   (both its fit cells carry anchor text).
6. **Reach.** Every reach >= 41/42 (all >= 0.95). The run tested follow transfer, not the reach factor (PREREG
   reading, Vera #28636).
7. **Timing and rules.** PREREG pushed 2026-11-23T07:04:44Z (commit 379c495b2, header 13810a2bc); first access-log
   line 07:07:34Z (A), 08:17:42Z (B thinking off), 08:20:15Z (B thinking on). PREREG.md unchanged since the header
   commit; no "RULE CHANGE:" line in DEVIATIONS.md.

## Where the miss comes from (`out_mech.txt`)
Qwen3-8B's 11 raw-HTML runs that saw the link and did not request it all answered by step 2-4; 7 gave the stale
2024 figure, the rest said the 2025 figure is not on the page. Its visible-link gate in raw HTML is 22/30 = 0.733,
the same as its hidden raw cell 31/42 = 0.738. So for Qwen the raw-HTML deficit is about the raw format, not about
hiding: "carries anchor text" is not one class for this model. [INFERENCE] raw HTML is longer and noisier than
markdown, and Qwen3-8B satisfices earlier on it.

## Limits of this rebuild
Arithmetic and event extraction on saved outputs; no second model run. My bootstrap uses my own RNG, so interval
ends differ from the build's in the 4th decimal.
