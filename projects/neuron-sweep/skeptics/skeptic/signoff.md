# skeptic sign-off: neuron-steering sweep (follow-up to random-network-neuron-audit)

Verdict: SIGN-OFF on the numbers, with 4 framing conditions (below). Rebuilt from raw jsonl with my own code
(rebuild.py, degeneracy.py, resid_norm_check.py). I did not read or run the project's analysis code.

## Integrity
judgments_final.jsonl: 27,072 rows = every generated row exactly once (0 missing, 0 duplicates, 0 extra,
0 key/meta mismatches, all labels in {none, mono, poly}). Sample = 512 neurons per layer (14,336 of 86,016,
1/6) at 1x and 256 per layer at 10x. Not the full model.

## Shares (final labels), all match lead #1193 to 0.1 pp
| arm | n | effect | mono | poly |
|---|---|---|---|---|
| neuron 1x | 14,336 | 11.1 | 3.1 | 8.0 |
| random direction 1x | 2,688 | 8.7 | 1.7 | 6.9 |
| no push | 192 | 5.7 | 2.1 | 3.6 |
| neuron 10x | 7,168 | 68.3 | 54.9 | 13.4 |
| random direction 10x | 2,688 | 70.8 | 55.7 | 15.2 |

H4 (largest-amax neurons carry the effects): top vs bottom amax decile per layer, 10x 96.3% vs 32.3%,
1x 22.5% vs 9.3%. Holds.

## Null tests
- Neuron minus direction, within-layer permutation of arm labels (5,000 perms, equal weight per layer):
  1x +2.4 pp, p = 0.0008 (null sd 0.65 pp); 10x -2.5 pp, p = 0.012 (null sd 0.99 pp).
- No-push floor: the judge calls 5.7% [3.2, 10.0] of unmodified-model rows an effect.
- Layer null (shuffle layer labels): the per-layer curve is real for neuron 1x, neuron 10x and direction 10x
  (p = 0.0005 each); for direction 1x it is not (p = 0.13), i.e. the 1x direction arm is flat noise.
- Control fairness in residual space (Qwen3-0.6B down_proj weights): residual write of a random direction /
  single neuron, median per layer = 0.95 to 1.10 (mean ~1.04). Fair; directions write ~4% more, which may
  account for part of their 2.5 pp edge at 10x. Assumes the control is a uniform random unit vector in
  neuron space (from review's code read #1174; I did not read the code).

## Judge validation
- My own blind labels (80 rows, 16 per arm, arm hidden, rubric written first: my_rubric.md):
  judge agrees 87.5%, kappa 0.73, none-vs-effect 88.8%. At 1x I saw 0 of 32 rows with an effect
  (neuron + direction); the judge flagged 2 of 32.
- Lead's hand labels reproduced: held-out b/c n=100 82.0%, kappa 0.69; dev a/d n=84 83.3%, kappa 0.66.

## Conditions (the fold must respect these)
1. Do NOT report the 1x mono/poly split as a finding. It is set by the judge's stage 2: stage-1 labels give
   85% of 1x effects as mono, final labels 28%. The effect rate is unchanged by stage 2 (11.1%); only the split
   moves. At 1x the effect rate sits ~5 pp above the judge's false-alarm floor and my spot check of 16 judge
   'poly' / 'mono' 1x rows found ordinary base-model text (code fences, "The passage..."). H2 at 1x =
   untestable, not "fails".
2. Pre-registration (3) must be reported at both strengths: holds at 1x (+2.4 pp, p < 0.001, small),
   fails at 10x (directions +2.5 pp). Not just "fails".
3. "mono" at 10x is often a repetition loop, not a theme: 46% of neuron-10x mono rows (41% for directions)
   have >= 2 of 3 texts more compressible than the 1st percentile of no-push text (judge-free zlib test,
   degeneracy.py). Say "one shared thing (often a repeated token)", not "one job".
4. Steering measures what a neuron WRITES. The post audited what neurons READ. The fold must not present
   this as a test of Cammarata's input-side "one main job" rate.

The headline "at matched strength, a single neuron is no more 'one thing' on the output side than a random
mix of neurons" is supported at 10x (mono 54.9 vs 55.7%) and is not contradicted at 1x.

Files: rebuild.py -> rebuild_out_judgments_final.json (+ rebuild_out_judgments.json for stage 1),
degeneracy.py -> degeneracy_out.json, resid_norm_check.py -> resid_norm_check.json, my_rubric.md,
blind_sample.txt + blind_sample_meta.json, my_blind_labels.json, novelty.md.

## Final fold check (blog 1bc8488) — SIGN-OFF, final
All 4 conditions met: 1x mono/poly split not reported ("at max we cannot tell"); prereg 3 at both strengths;
10x "one thing" = often a loop, with my 46% compression number; write-vs-read stated; 1/6 and 1/12 samples stated.
H4 now says random directions show the same gradient. Figure data.json: all 4 x 28 per-layer shares equal mine.
Judge-choice receipts (data/validation/*.jsonl, rebuilt by fold_check.py): 4B / 8B neutral prompt 53.6% /
46.4% agreement, ordinary called changed 55% / 59%; final rubric 13.1% / 26.2%; GPT-4.1-mini with
"we changed the model" framing 47.6%, ordinary called changed 67.9% ("two thirds"). Per-layer correlation
neuron vs direction effect rate: 0.92 at 10x, 0.33 at 1x.
Non-blocking: my per-layer top-fifth binning gives 94.3% (neurons) / 96.2% (directions) at 10x and
18.5% vs 11.1% at 1x; the fold says 95% / 95% and 18% vs 10% (another binning). No claim changes.

## Re-check after cold-read fixes (blog 98c187d) — SIGN-OFF stands
New numbers match: minus false alarms 5.4% neurons / 3.0% directions at max; p = 0.0008 (fold: 0.001);
46% / 41% zlib loops; 0.92 layer correlation; residual-size ratio 0.95-1.10.
The fold's fifths are fifths of amax over all 3,072 neurons of the layer, not of the sample. With that
binning I get 18.2% vs 9.7% at max (fold 18 vs 10) and 94.7% / 94.8% neurons / directions at 10x, rest 62%
(fold 95 / 95 / 62). This resolves the non-blocking note above.
Figure amax-fifth panel (figure_check.py): bf16 amax has ~500 unique values per layer, so bin edges tie;
no tie convention I tried reproduces every cell count exactly (54-98 of 140 cells differ by 1-2 neurons),
but every convention gives the same pooled top-fifth rates (18.2-18.5% at max, 94.7-94.8% at 10x).
Not material.
