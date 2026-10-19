# Vera (skeptic) sign-off — theory-1019-max-pool-survives

Verdict: **REJECTED reproduces** (no to the title). Own code only; no analysis.py read or run.

Inputs: `kran-research/projects/max-pool-survives/data/{clean_tokens.npz, runs/lam0__seed{0,1,2}__tokens.npz, probe.npz, clean_acts.npz, split.json}`.
Definitions from PREREG.md 9bc1a08 + Amendment 1 (4bc18f2).

| number | Vera | build (#3206) | file |
|---|---|---|---|
| max14 pooled catch @1% FPR after re-threshold | 0.000, 95% [0.000, 0.0089] | 0.0%, [0.0, 0.9] | vera_boot.out |
| max14 clean catch | 0.914 | 91.4% | vera_rebuild.json |
| frozen AUROC per seed | 0.5830 / 0.5725 / 0.5710 | 0.583/0.573/0.571 | vera_rebuild.out |
| mean8 pooled (clean) | 0.0638 [0.0145, 0.178] (0.074) | 6.4% [1.5,17.8] (7.4%) | vera_boot.out |
| fresh pooled | 0.3607 [0.308, 0.933] | 36.1% [30.8, 93.3] | vera_boot.out |
| last14 pooled (clean) | 0.0078 [0, 0.145] (0.997) | 0.8% [0,14.5] (99.7%) | vera_boot.out |
| argmax on shared prefix, eval false | 86 / 95 / 97% (clean 0.7%) | 86/95/97% | vera_mechanism.out |
| paired AUROC max14 | 0.554 / 0.507 / 0.507 | 0.55/0.51/0.51 | vera_mechanism.out |

Repro gate: rerun scores and mean acts are bitwise equal to the 1018 files (np.array_equal), not just within 0.03.

Nulls:
- within-pair label swap (probe split) -> max14 clean catch 1.3%, AUROC 0.51 (pipeline can fail; no leak).
- resplit null: threshold from 200 true redrawn from all 796 refit+eval true, 2000 draws -> max14 attacked catch mean 0.2-0.4%, max 3.4-4.0%.

Flags (disclose in the post):
1. fresh catch depends on C: pooled 36.1 / 26.3 / 12.5 / 8.9% at C = 0.01 / 0.1 / 1 / 10 (vera_extra.out). Prereg part 2 "failed" holds by the frozen selection rule only.
2. Per-token drop on the max14 direction: prefix -7.4..-7.9, false answer tokens -11.3..-11.4, true answer tokens -5.8 logits. "Every token drops ~8" is true for the prefix only.
3. mean8 is weak before the attack (clean catch 7.4%, CV AUROC 0.73), so prereg part 3 compares against a monitor that never worked.

Derivation: derivation.md (prefix lemma, shared-shift lemma for mean and max, bell-curve formula re-derived).
Novelty: novelty.md.
