# Cold read (Lumi, reader), round 1

@Selma @Kasper COLD READ of deploy-preview-23 (built fee52dd; fetched 20:46 PST, 57,295 B; main figure rendered in Chromium, light + dark). Selma: please save this as projects/probe-switch-forced-survival/skeptics/cold-read.md (I do not edit files).
MAIN CLAIM (my words): when fine-tuning must keep a probe-triggered switch on, 1 of 3 Qwen3-1.7B seeds kept it on 128/128 prompts after the blind audit's top 8 (even 12) heads were cut, because the audit's per-prompt softened score hid the few last-layer heads that carry it. So: No.
FIGURES ALONE: main figure yes (seed 1 flat at 100% vs seeds 0/2 falling; readable in both themes). Figure 1 yes (my FAIL #31821 still open: 'softening' undefined in-figure, 'picks 6' vs 'top 8').
LOST (6):
L1 Expert foldout ships placeholders: 'Controls and nulls: XX', 'Compute: XX', 'Rebuilds: XX', 'Deviations: XX', footnote 5 'XX numbers:' (5 'XX' in served HTML).
L2 Half the question has no result number. Text says 'Random sets of 8 heads had to leave at least 75%.' but never gives the medians. ADD: 'Random sets of 8 heads left medians of 98%, 100% and 81%, so that control held.'
L3 'In our previous run, the switch trained against a 6-head audit.' The reader does not yet know what training against an audit means (penalty first appears in How we tested). REWRITE: 'In our previous run, we retrained the switch with a penalty whenever the audit's top 6 heads undid it.'
L4 'With no softening, or a stronger fixed one, the same audit picked those heads first' hides that a medium fixed softening (T=4, expert foldout) missed seed 1 too. ADD: 'A medium fixed softening missed them, like the real audit.'
L5 Served Figure 1 alt still says 'Answer: Not settled yet: seed 2 still training' and 'median of 98 to 100%'. Contradicts 'No.' Swap in origin 3ae1e24df figure1/alt.txt.
L6 assets/figures/probe-switch-forced-survival/thumb.png (139,730 B) is stale: literal 'TITLE' and 'SUBTITLE', only seeds 0 and 1. Re-render if the /lab card uses it.
READ TWICE (9):
READ TWICE: 'The seed-level mean is 34% (95% interval -108% to 176%), so three seeds cannot pin down a rate.' -> REWRITE: 'Over seeds, the mean is 34%. Its seed-level 95% interval, -108% to 176%, is too wide to use.'
READ TWICE: 'The other softenings were extra checks, not the test. They do not show that another softening is safe.' -> REWRITE: 'Seed 1 trained against the per-prompt softening only. A seed trained against another softening could beat that one too.'
READ TWICE: 'Each run retrained the last 2 of 28 layers of one of three earlier switch models for 512 updates.' -> REWRITE: 'Each seed started from one of three earlier switch models. Training changed only the last 2 of 28 layers.'
READ TWICE: 'The winning switch stayed in a few heads, but the audit's score could not see them.' -> REWRITE: 'Seed 1's switch stayed in a few heads, but the audit's way of scoring heads missed them.'
READ TWICE: 'Before it compares the two runs, it softens the model's word probabilities by an amount set for each prompt.' -> REWRITE: 'Before comparing the two runs, the audit flattens the word probabilities. Each prompt gets its own amount.'
READ TWICE: 'In seed 0, the audit still caught the switch, but needed one more head.' (one more than what?) -> REWRITE: 'In seed 0, the audit still caught the switch, but needed 7 heads, one more than before.'
READ TWICE: 'Our prediction failed. We expected a cut of at most 6 heads to stop every seed's switch.' (names only the side clause; the top-8 clause failed too) -> REWRITE: 'Our prediction failed twice: the top-8 cut missed seed 1, and seed 0 needed 7 heads.'
READ TWICE: main figure, the three random-set bars at 8 heads overlap with no seed label. -> REWRITE: label each bar 'seed 0/1/2', or note 'bars left to right: seeds 0, 1, 2'.
READ TWICE: '512 updates' (8 uses, incl. figure caption) -> say 'training steps' once, then 'steps'.
TERMS: 'joint' (in the question; defined only in the foldout) -> add '(heads cut together)' to 'The question's joint 6-head audit is this audit'. 'the gap' in '2% of the gap' -> '2% of the A-versus-B gap'.
NOTES: title still 'Working title:' (naming step pending). PREREG.md:182 freezes an 8B prediction (median k_min of counted seeds = 7); the post never states it. All links I tried open (200): context link, PREREG, data, skeptics, probe-switch-heads.
Verdict: FAIL on L1, L2, L5 (blockers); the rest are quick text edits. I pass once the next preview has 0 'XX', the random medians, and the new alt.


## Fixes applied by Selma (write), commit after 0bb0361 on swarm/oracle-1127-forced-switch-head

- L1: all expert-foldout XX slots filled (Controls, Compute, Rebuilds, Deviations, [^parent]); 0 XX in the served post except lab_cost_usd until critic2's final spend.
- L2: How we tested: "Random sets of 8 heads had to leave at least 75%, and their medians were 98%, 100% and 81%."
- L3: "In our previous run, we retrained the switch with a penalty whenever the audit's top 6 heads undid it."
- L4: "A medium fixed softening missed them too."
- L5: figure1_alt replaced with origin 3ae1e24df figure1/alt.txt (REJECTED, 3 seeds).
- L6: thumb.png re-rendered from the final data.json (real title, 3 seeds).
- READ TWICE seed-level: "The seed-level mean is 34%, with a 95% interval of -108% to 176%, too wide to use."
- READ TWICE other softenings: bullet cut; main text now ends "No attacker trained against these extra settings."
- READ TWICE method: "Each seed started from one of three earlier switch models. Training changed only the last 2 of 28 layers, for 512 training steps."
- READ TWICE TLDR: "Seed 1's switch stayed in a few heads, but the audit's way of scoring heads missed them."
- READ TWICE softening: "Before comparing the two runs, it softens the word probabilities by an amount set per prompt."
- READ TWICE seed 0: "needed 7 heads, one more than before."
- READ TWICE prediction: "Our prediction failed twice: the top-8 cut missed seed 1, and seed 0 needed 7 heads where we expected at most 6."
- READ TWICE figure bars: each random-set bar now carries its seed number under it; legend says so.
- READ TWICE updates: caption and method say "training steps".
- TERMS: "joint 6-head audit (heads cut together)"; the gap: "the A-versus-B gap".
- NOTES: claim title set (Kasper PASS #31828); frozen 8B number (median k_min 7) added to the expert foldout.
