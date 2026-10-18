# prior_70m: re-threshold recovery in an earlier Lab attack (exploratory)

Not pre-registered. A reanalysis of saved scores from room theory-1017-fisher-pullback
(post: https://blog.kran.ai/fisher-pullback-transfer/), branch `swarm/theory-1017-fisher-pullback` @ c29609a.

That attack fine-tuned blocks 0-2 of Pythia-70M against one frozen probe (layer 3, mean over
positions 1-63), with a softplus loss on texts with the concept and an output-KL retain loss
(weight 30; plus one run per target at weight 0). It stopped at the first check (every 10 steps)
where the target probe's test AUROC fell below 0.6. 14 of 15 weight-30 runs stopped at step 10.

For each run's target probe, `refit_recovery_70m.py` reports AUROC clean/attacked, TPR at 1% FPR
(threshold = 99th percentile of all test negatives), the attacked TPR at the clean threshold and at
its own threshold, the mean score shift of negatives and positives in clean-negative SDs, and SD ratios.

Run from the kran-research clone (reads the old branch with `git show`):

    uv run --with numpy --with scikit-learn python projects/rethreshold-probe-evasion/prior_70m/refit_recovery_70m.py

Result (`refit_recovery_70m.out`): on the two targets with a large drop (dbpedia:NaturalPlace,
lang:it; clean TPR 0.97-0.99, 8 runs), the re-fit threshold won back 0.00 of the lost TPR; AUROC fell
to 0.15-0.51. Negatives moved down with the positives (common-mode), but the classes also lost or
inverted their order. On the three weak targets (clean TPR 0.09-0.31) the shares (0.02-0.78) rest on
1-7 points of TPR and are noise-level.
