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

## B. Map values (J0, untrained Qwen3-8B, eager tail on cached block-28 inputs)

To be written from research_checks/map.json and research_checks/route.json after J0, before the first MLP-8 /
last-MLP job is submitted.
