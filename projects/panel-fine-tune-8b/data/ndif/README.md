# NDIF clean-gate cross-check (Iris; never the status)

Clean Qwen3-8B, read remotely on NDIF (NNsight and NDIF, arXiv 2407.14561; bf16, 47 traces of 64 statements,
2026-11-16 ~06:03 CET). Country-token states at the 20 PREREG s2 readouts (19 pre-norm block outputs via
`model.model.layers[R-1].output`, R = 37 via `model.model.norm.output`), stacked server-side.
Probes fit locally with the project's rule (`refit.select` / `refit.fit`: StandardScaler + logistic regression, C by
city-grouped 5-fold out-of-fold AUROC), z-moments on the 700 fit-pool true statements, line = 2nd-highest of the 200
`refit`-true statements. Script: `iris_checks/ndif_clean_gate.py`. Raw states (468 MB fp16) are not committed:
`cache_sha256.txt` lists the batch files.

Result (`ndif_gate.json`): max-panel catches 596/596 eval false statements (100%) at its own line (3.70), eval false
alarms 8/596 (1.3%). Readouts 15-37 each catch 100% alone; 13: 82%, 14: 96.6%. Every probe chose C = 0.01; OOF AUROC
0.988 (R13) to 0.9999. The PREREG s6 gate (>= 90%) is the laptop job A's clean score, not this file.
