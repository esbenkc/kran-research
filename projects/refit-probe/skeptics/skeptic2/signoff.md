# Hugo (skeptic2) sign-off — refit-probe

**Answer to the title: NO (REJECTED), reproduced. Pre-registered prediction: FAILED, reproduced.**

Own code only (`check.py`, `verdict2.py`, `clean_train_split.py`); I did not read `refit.py` or another skeptic's scripts.
Inputs: `data/answers_*.jsonl`, `labels_*.jsonl`, `acts_*.npz` (own_L12, base_L12), `probes.json` (commit cb67be2).

## Data checks (check.py:80-104)
- 400 test prompts identical and in the same order in all policies; rows 0-399 are the test rows.
- 0 test prompts in any train pool; 0 duplicate prompt ids; 0 overlap with the 398 prompt ids of probe-ensemble DPO train/eval/fixed files.
- Frozen probe A rebuilt from probes.json on own_L12: max |p diff| 4.9e-4 (float16), flagA agreement 1.000 in every policy. The acts are what the schema says.

## Key numbers (n=200, class-balanced draws, StandardScaler + LogReg C=0.05, mean AUROC over 20 draws, 400 test answers, OTHER dropped)
| policy | refit own | refit held text (base reads) | old probe | build (refit) |
|---|---|---|---|---|
| base | 0.818 | 0.818 | 0.762 | 0.822 |
| lam1_s0 | 0.803 | 0.803 | 0.776 | 0.794 |
| lam1_s1 | 0.776 | 0.776 | 0.768 | 0.773 |
| ens_s0 | 0.763 | 0.763 | 0.749 | 0.768 |
| ens_s1 | 0.723 | 0.721 | 0.626 | 0.710 |

Verdict gaps (base minus policy), paired bootstrap over the 400 shared test prompts, 1000 resamples, 4 independent draw seeds (check.py:177-205, verdict2.py):
- lam1_s0 0.008-0.020 (CI lower < 0)
- lam1_s1 0.035-0.046 (CI lower < 0)
- ens_s0 0.050-0.060 (CI lower -0.004 to +0.008: borderline, on the 0.05 line)
- **ens_s1 0.089-0.113, CI lower 0.036-0.058 in 4/4 seeds -> NO** (build: 0.112 [0.060, 0.169])

Guards (PREREG s8): judge-vs-haiku agreement base 0.864, lowest ens_s1 0.820 (drop 0.044 < 0.05: no label guard). Robustness subset (TEST answers where judge == string check, refit trained on judge labels): ens_s1 0.069 [0.010, 0.128] -> NO, same as main.

Prereg: refit AUROC at n=200 is 0.72-0.80 on the lambda=1 policies (all < 0.9); true/false-facts probe B2 (layer 16) 0.36-0.44 (< 0.8, below chance). FAILED on clause (i).

Catch at 5% false alarms (threshold on test honest answers), n=200 refit: base 0.42, lam1 0.34/0.32, ens 0.26/0.24. Of the lies the old probe misses at its old threshold: 0.32/0.25 (lam1), 0.13/0.21 (ens).

## Nulls (check.py:152-168, n=200, 10 draws)
- Shuffled train labels: 0.49-0.53 on every policy.
- Random directions on own_L12: range 0.36-0.66 (10 draws per policy).
- Bag-of-words LogReg on the same 200 answers: 0.58-0.63; answer length only: 0.52-0.57.
So the refit is real signal (well above the nulls), and well above a word-count reader.

## Caveats the post must respect
1. The NO rests on ONE of four runs (ens_s1). ens_s0 sits on the 0.05 line and flips with the draw seed; do not say "the ensemble runs".
2. **Sensitivity outside the prereg (clean_train_split.out):** if the refit is ALSO trained only on clean labels (judge == string check), clean-test AUROC rises to base 0.914, lam1 0.927/0.903, ens 0.900/0.869, and the ens_s1 gap falls to 0.045 [-0.007, 0.099] (verdict2_out.json `clean_train_and_eval`), which would be YES. The base clean test set has only 38 honest answers. The prereg robustness subset is test-only, so the status stays NO, but the NO depends on judge-label noise in the training labels. One foldout sentence should say so.
3. Held text equals own text within 0.002 on every policy: the separation the refit finds is the same whether the trained model or the untrained model reads the answers. The gap to base sits in which answers the policy writes, not in degraded activations. The prereg's if-wrong reading ("signal really degraded at the probed layer") is not supported in that form.
4. A new threshold on the old probe (its AUROC) is within 0.03 of the refit on 3 of 4 runs; the refit is only clearly better on ens_s1 (+0.10 in my rebuild, +0.08 build).

## Follow-up on caveat 2 (spin_lies.out)
Rebuilt Vera/Iris/Kasper's point with my own code: of the judged lies on the 400 test answers, the share where the string check says honest (the answer states the fact and then spins it) is base 0.109 (33/303), lam1 0.175/0.109, ens_s0 0.221, ens_s1 0.374 (70/187). The second judge (Haiku) calls 0.914 of ens_s1's spin lies deceptive (n=70; base 0.939, n=33). So the agreed-labels filter mainly removes real lies, a third of them in the run that drives the NO. The clean-only YES-side reading drops the hard lies; it is not evidence that the NO is label noise. Verdict stays NO.
