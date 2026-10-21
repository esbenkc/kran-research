# Hugo handoff — oracle-1021-refusal

- Claimed task: `skeptic2 independent rebuild`; reviewer Tove.
- Novelty notes are in `swarm-oracle-1021-refusal/skeptic2/novelty.md`.
- Base rebuild from `runs/base/test.judged.jsonl`: harmful refusal 192/200 = 0.960; harmless over-refusal 12/200 = 0.060.
- D7a status bar to rebuild: new harmless refusals vs base-compliant TEST harmless rows, pass iff <=9/200.
- `checker.py` implements base rates, D7a paired harmless join, chosen-steered status when `--steered-test` is provided.
- `base_rates.py` and `base_rates.json` record the base bar problem and flagged harmless prompts.
- `readout_nulls.py` rebuilds base-direction AUROC from `acts.npz` and runs random/shuffled nulls.
- Partial readout result: stripped base-direction AUROC peaks 0.983 at L12; shuffled-label P95 0.924 and max 0.963 at that layer.
- Remaining blocker: wait for Bruno to commit DEV pick (`picked.json`) and seed-0 steered TEST judged files.
- Final verdict must cite my checker output path and answer the title only after raw TEST + null data rebuild.
