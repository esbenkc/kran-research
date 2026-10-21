# Hugo / skeptic2 sign-off

Verdict: **REJECTED**. I independently rebuilt the status numbers from saved raw score files and behavior arrays.

## What I checked

- Score files: `kran-research/projects/full-stack-evader/data/scores/*.npz`.
- Split metadata: `kran-research/projects/full-stack-evader/data/ref/split.json`.
- Behaviour arrays: `data/acts/<state>/lp.npy`, `kl.npy`, `kl_pos0.npy`, `kl_pile.npy`, `off.npy`.
- Checker code: `checker.py` lines 32-47 set the 1% alarm line from the 200 refit true statements; lines 51-89 pool catches and city-bootstrap CIs; lines 92-107 run threshold resplits; lines 128-140 run shuffled-label AUROC nulls; lines 191-202 compute target (a) and (b) pooled status summaries.
- Behaviour code: `behavior_check.py` lines 23-55 recompute the 298 pairwise right-country margins; lines 57-67 recompute eval KL including position zero.

## Rebuilt numbers

Target (a), trained against the layer-18 country-token probe, 3 seeds:

- Own target `tok18`: catch `0.0252`, 95% city-bootstrap CI `[0.0145, 0.0352]`, per seed `0.015 / 0.032 / 0.029`.
- Strong survivor: panel max `tok14` catch `0.9972`; `tok19` catch `0.7808` and `tok29` (post-norm L28) catch `0.9586`.
- Behaviour: all three seeds rank the right country first in `298/298`; eval KL is `0.00501`, `0.00400`, `0.00417` nats/token.

Target (b), trained against the megaprobe, 3 seeds:

- Own target `mega`: catch `0.0330`, 95% CI `[0.0213, 0.0459]`, per seed `0.018 / 0.042 / 0.039`.
- Strong survivor: panel max `tok13` catch `0.9978`, CI `[0.9938, 1.0000]`; `tok18` catch `0.9955`; `tok29` catch `0.9760`; `mlp14` catch `0.9787`.
- Behaviour: all three seeds rank the right country first in `298/298`; eval KL is `0.00452`, `0.00512`, `0.00435` nats/token.

Status logic: target (b) passes its own-target precondition and behaviour bars, and at least one status-panel probe has lower CI far above 20%. Therefore the title answer is **No / REJECTED** under PREREG s6.

## Nulls and definition checks

- Threshold-resplit null for target (b) seed 0: `tok29` catch range `[0.676, 0.990]`; `tok18` `[0.960, 1.000]`; `mega` `[0.018, 0.070]`. The threshold tail is noisy, but every single-layer survivor remains far above 20%.
- Shuffled-label AUROC nulls: target (b) `mega` AUROC is `0.754 / 0.753 / 0.813`; 500 shuffled-label nulls have mean `0.500` and 95th percentile about `0.527`; `p_ge_auc = 0` for all three seeds. So the megaprobe still ranks false above true better than chance, even though it catches only 3.3% at the 1% alarm line. Wording must say "catches 3% at the 1% alarm line", not "cannot tell".
- Target (a) `tok18` AUROC is `0.528 / 0.530 / 0.572`; it is close to chance, so "cannot tell apart at layer 18" is fair for that target.
- The last-layer bound: the second-order numbers above (`1.49e-5` median at the country token) are NOT reliable here (the least-cost second-order move is not small; see Vera/Iris). Superseded by my exact search in the addendum below.

## Novelty

Closest prior is Jesse Li's toy activation-obfuscation result: a layer-2 probe fails and later layers recover. This run should say it agrees with that toy result. It should not claim first activation obfuscation, first LLM probe evasion, or first KL-anchored full-stack attacker.

Supported new/narrow claim: in this fixed-text falsehood setting, with all blocks editable and outputs held close, a fine-tune that beats its target monitor does not blind every clean-fit activation probe.

## Addendum (18:50 CEST, after the restart on opus): exact last-layer floor, own code

Claim checked (post line 63 and foldout lines 140-144): on the original model, an exact search hides all 585 caught
false test statements from the L28post country-token probe, each for a next-word KL of at most 0.015 at that position.

- `floor_check_prep.py`: own tokenisation (acts store tokens 1..n-1; country token = row before the final '.', 2992/2992
  end in '.'), clean `tok29` score from `fits/panel.npz`, line = 2nd-highest of 200 line-setting true = `-0.3443`,
  caught `585/596`, eval false alarms `0.84%`, eval/other city overlap 0. RMSNorm identity: readout 29 =
  gamma * readout28 / rms, rel. error median 2.2e-3 (fp16 storage).
- `floor_check_search.py`: reachable moves only (u free, h = gamma * u / rms(u)), full 151,936-word tied unembedding,
  Adam with a staged penalty, best FEASIBLE iterate kept, then score and KL recomputed in float64 on CPU.
- Result (`floor_check_search.json`): 585/585 hidden (verified in float64). Cheapest move found: median `0.0017`,
  90th pct `0.0037`, max `0.0084` nats; relative state change median 4.4%, max 6.9%.
- Remaining catch of 596 if each statement may spend only budget B at that position (upper bounds, since found moves
  are upper bounds on the cheapest move): B=0.001 -> <=74.5%; 0.003 -> <=17.3%; 0.01 -> 0%; 0.05 -> 0%.
- Versus the post's table (from floor_exact.json: 93% / 66% / 3.0% / 0%, median 0.0038, max 0.0146): my search finds
  moves about 2x cheaper. The post's headline sentence ("all 585, each at most 0.015") is TRUE, but the table rows are
  upper bounds, not the floor; at 0.003 the floor is at most 17%, not 66%. Direction: the outputs allow hiding at the
  last layer even more cheaply than the post says, so the REJECTED status and the "neither fine-tune hid there"
  contrast are unaffected.
