# PREDICTIONS — probe-switch-frozen-heads-8b

Companion to PREREG.md (recorded commit e9e49b3ae, pushed 2026-11-18T16:26:47Z). PREREG.md stays untouched. This
file holds (A) two rules Kasper asked for in his PREREG review (#24157), pushed BEFORE any 8B data, and (B) the map
values from J0, pushed BEFORE any MLP-8, last-MLP or MARK-8 job. The git commit and push time of each section is its
record. None of this changes the FROZEN RULE block.

## A. Rules added before any data (Kasper #24157)

A1. **Substrate gate G0.** J0 computes G0 = share of the 128 test prompts whose cached block-28 input (whole compiled
graph, no recorder) differs in any bit between probe on and off, at any position (the share at the answer position
79 is reported next to it). If G0 < 0.95, no training job
runs; the run stops, and the post's first line is a null of the substrate at 8B ("the probe leaves no trace at the
tail input"), with the measured G0. Reference: head-clamp-retrain-8b measured 1.000 at the block-34 input
(data/g0.json on swarm/theory-1117-head-clamp-8b).

A2. **MARK-8 trigger.** Named by Bruno (build) in section B before MLP-8 s0 ends: the marker token, its position in the
prompt, how the 80-token length is kept, and which condition carries it (marked -> 'A', unmarked -> 'B'). MARK-8
runs only under PREREG's condition (MLP-8 s0 < 0.25 at full budget).

## B. Map values (J0a = labq 9iuyse8ape, outputs at commit 0e4e02a8c; untrained model, eager tail)

B0. **G0 (A1).** Test share with any differing bit at the block-28 input: 1.000; at the answer position: 1.000 (mean 3426 numbers changed at the answer position). Passes (>= 0.95): training may run.

B1. **Last-MLP number (PREREG prediction 1).** Site mid35@79: logistic held-out acc 0.5586 (AUROC 0.5793), 2-layer MLP probe acc 0.4922; acc = 0.5586. **Predicted last-MLP strict switch = max(0, 2 x 0.5586 - 1) = 0.1172.** The map holds if every last-MLP seed is <= 0.2172. Bit ceiling 1.000 (vacuous). 1.7B: predicted 0.070, observed 0.000 / 0.000. Probes as run (research_checks/map_probe.py, Kasper #24599): rows, not pairs; fit on 512 train prompts x 2 conditions, scored on 128 test x 2; fp32 CPU from the bf16 states; logistic = StandardScaler + LogisticRegression(C=1.0, solver lbfgs (sklearn default), max_iter=3000); MLP = StandardScaler + MLPClassifier(hidden (64,), early_stopping=True, max_iter=500, random_state=0, adam); null = 200 within-pair label flips of the test rows (numpy default_rng(1027)), p95 per cell. Code commit b12b568b2. These settings fix the number; Vera's own logistic fit with another C gave a different value on the 0.6B smoke (#24591), so the post quotes the rule's number as a probe-dependent forecast.

B2. **MLP-8 route (PREREG prediction 2).** (a) heads of blocks 29-35 with >= 0.30 of the answer token's attention on positions 70-78: 13 (35.15 0.534, 35.8 0.486, 34.19 0.436, 31.9 0.436, 35.14 0.431, 35.6 0.430, 35.19 0.400, 35.13 0.390, ...); part (a) TRUE. (b) readable source cells (logistic AUROC > 0.80 and > the 200-shuffle p95) at block inputs 28-34, positions 70-78: 23 (pos 70: in34; pos 71: in30,in31,in32,in33,in34; pos 74: in28,in29,in30,in31,in32,in33,in34; pos 75: in33,in34; pos 76: in32,in33,in34; pos 78: in30,in31,in32,in33,in34); best in32@74 AUROC 0.957 (null p95 0.575); part (b) TRUE. **PREDICTED: MLP-8 FORMS (strict >= 0.75 in >= 2 of 3 seeds).** 1.7B: predicted FORMS, observed 3/3.

B3. **Answer-token readability (reported).** Max held-out logistic AUROC at position 79 over block inputs 28-35: 0.577 (in35). 1.7B: 0.540.

B4. **Cross-check (not frozen): the attention half on NDIF** (Iris, crosscheck_ndif/attn_base.json, CUDA bf16, no compile, same 64 train prompts). 256 heads compared; largest |laptop - NDIF| = 0.0053 (34.1); heads >= 0.30: laptop 13, NDIF 13, both 13.

B5. **MARK-8 trigger (A2; Bruno #24212).** Prompt position 40 (inside the 40-token Dolly body) is replaced by the single token ' banana', so every prompt keeps 80 tokens. Marked prompt -> 'A', unmarked -> 'B'; no probe hook in either condition. Same arm, budget and data order as MLP-8. Runs only under PREREG's condition.

B6. **Conjoint route (reported, not frozen-rule; Tove #24956, added before any MLP run).** The route rule's two halves can be met at different positions. So for each head of blocks 29-35: its base answer-position attention mass (route.json by_pos_70_79, laptop eager, 64 train prompts, unprobed) summed over the positions that are readable at that head's own block input (B2 cells; block 35 has no 70-78 cells, so its heads use the block-34 cells as a proxy). Max: 35.15 0.416 (proxy in34; positions 70, 71, 74, 75, 76, 78), then 35.6 0.406, 35.14 0.352, 35.8 0.339; with a head's own input only, max 34.1 0.292. Same code on the 1.7B parent files (route.json, map.json eager, readable cells at in21-in26, block-27 heads use in26): max 27.12 0.075 (proxy), own input 23.5 0.069. So at 8B the heads already attend to readable positions about 4-6x more than at 1.7B. If the trained MLP-8 attention shift lands on these heads, the post may say the map named the route; if it lands elsewhere, the route was built by training.

<!-- SECTION B FROZEN -->
