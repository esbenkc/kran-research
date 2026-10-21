# Vera (skeptic) sign-off — oracle-1021-refusal

Title: "Does adding back the refusal direction at one layer restore a stripped model's refusals without over-refusing?"

**SIGN-OFF: REJECTED (No to the title).** 0 of 3 LoRA seeds pass PREREG s8 at the DEV-frozen point (L*=15, k*=1.5).
The No rests on condition 1 (harmful refusals restored), which fails by about 40 points on every seed and with
both judges. Condition 2 (new harmless refusals) fails on all seeds with the build's judge (11/16/15), but with my judge
it is 4/4/10 (passes on seeds 0 and 1), so it must not carry the No alone.

All numbers below come from my own code on the build's saved data (and, for D10, my own model run).

## Status (PREREG s8 + D7/D7a), TEST, n = 200 harmful + 200 harmless per arm

Base unsteered TEST: harmful refusal 192/200 = 0.960 [0.923, 0.980]; harmless 12/200 = 0.060 (10 refusal + 2
'incoherent'); 188/200 harmless answered. Bar: harmful >= 0.768; new harmless refusals <= 9.

| seed | harmful refused at (15, 1.5), build judge | my judge | new harmless, build judge | my judge | random dirs (build judge) | own refit dir |
|---|---|---|---|---|---|---|
| 0 | 75/200 = 0.375 [0.311, 0.444] (0.39 of base) | 0.245 | 11 | 4 | 0.075 / 0.050 / 0.105 | 0.040 |
| 1 | 64/200 = 0.320 [0.259, 0.388] (0.33 of base) | 0.190 | 16 | 4 | 0.065 / 0.070 / 0.100 | 0.040 |
| 2 | 73/200 = 0.365 [0.301, 0.434] (0.38 of base) | 0.220 | 15 | 10 | 0.075 / 0.045 / 0.085 | 0.060 |

Files: vera_status_build.out, vera_status_s1_build.out, vera_status_s2_build.out (build labels);
vera_status_myjudge.out, vera_status_s1_myjudge.out, vera_status_s2_myjudge.out (my Anthropic-family judge).
Code: vera_status.py (conditions at lines computing c1/c2/c3), vera_pick.py.

DEV pick rebuilt (vera_pick_build.out): L*=15, k*=1.5 (18/64 harmful refused, 3/61 new harmless), L_mid = 15;
all 28 k*_L match runs/picked.json. Base DEV harmful 62/64, so the DEV target was 49.6/64.
Timing: config.yaml (judge) 7ae37d4 at 17:01:49 PDT < first DEV output 17:54:00; picked.json 5493c54 at
18:35:26 PDT, the steered TEST step started the same second after the commit (logs/chain.log).

## Prereg outcome (seed 0): FAILED

- Part A (readable, AUROC >= 0.9): holds. Stripped model, base direction: 0.983 at L12 (vera_readout_strip.out).
  Null: 50 random directions give P95 0.790 (max 0.842) at L12, so the base direction is clearly above chance in
  the stripped model. In the BASE model the null is weak past layer 14 (random P95 0.92-0.97 at L16-22), so
  'AUROC >= 0.9' alone is not evidence; the post must show the random-direction reference.
- Part B (mid-layer restore): fails. L_mid = 15: 0.39 of base, 12/200 absolute harmless (literal bar < 5%).
- Part C (early/late < 30% of base): holds; max early/late is 0.104 of base (L23).

## Null and robustness tests (mine)

1. **Gain-grid artefact (D10, my own model load, direction and hook; vera_steer.py, vera_finegrid.py).**
   DEV at L15, 17, 18, 19 x k in {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5}, my judge. The best harmful
   refusal with <= 3 new harmless refusals is 14/64 (L18, k=0.6) against a target of 49.6/64. Harmful and new
   harmless refusals rise together at every layer (e.g. L18: k=0.8 36 (19 new), k=1.0 53 (40 new)). So the No is
   not a coarse-grid artefact. Hook check: at shared cells my refusal counts match the build's outputs
   (L18 k=1.0: 53 vs 53; L17 k=1.0: 33 vs 35); 1049/1536 generations are text-identical (batching differs).
   Job: 488 s on MPS, peak footprint 5026 MB, rss 726 MB (vera_finegrid_mem.log), one model process.
2. **Random directions at the same norm** (build's arms, my recount): 0.045-0.105 harmful refused, so the 0.32-0.375
   at L15 is specific to the base direction; condition 3 passes on every seed.
3. **Read-out null** (vera_reader_null.py): the DEV-fixed cut on 100 random directions in the stripped model catches
   median 7.0% / p95 42.6% / max 57.0% of TEST harmful at L15 (base direction: 62.0% at 3.0% harmless) and
   median 7.3% / p95 30.2% at L16 (base direction: 86.0% at 4.0%).
4. **Second-family judge**: my claude-haiku-4-5 judge (letter labels) agrees with gpt-4.1-mini on 0.955-0.985 of
   unsteered rows, but is stricter on steered harmful outputs ('Sure, ... then no content'): 0.245 vs 0.375 at
   (15, 1.5) seed 0. This moves the No further from the bar, and it moves condition 2 to a pass on seeds 0 and 1 (4, 4 new; seed 2: 10).

## Definition checks

- Harmless bar: the frozen absolute '< 5%' would fail the untouched base model (6.0%, 12/200). D7/D7a replaced it
  with the paired 'new harmless refusals <= 9', logged before any steered output (I raised it, #4056). The post
  must say both, and the prereg outcome is judged on the literal bar.
- 'Incoherent' harmless outputs counted as over-refusal include 2 wrong-but-fluent base answers. Immaterial to the No.
- The refusal direction is taken at the last prompt token (post-instruction, as in Arditi et al.), not at the
  instruction tokens, so it is the refusal-side direction in the sense of arXiv 2507.11878.

## Novelty: see novelty.md (no STOP; Lee et al. 2610.00320 App. H is the one prior write-back data point).
