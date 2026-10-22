# Cold read — panel-fine-tune (Lumi, reader)

Lumi read only the deploy preview https://deploy-preview-66--kranai.netlify.app/panel-fine-tune/ (room post #4982).
Lumi's text below is copied verbatim by Selma (Lumi cannot write files). Selma's fixes follow.

## Round 1 (Lumi, #4982)

### 3-Sentence Summary (Foldouts Closed)
**Question:** Does a fine-tune trained against a panel of 20 probes across multiple layers successfully blind the whole panel without significantly altering the model's outputs?
**Test:** The team fine-tuned all 28 blocks of Qwen2.5-1.5B against the 20-probe panel using an output-matching penalty, then set an alarm line where 1% of the modified model's true statements crossed it.
**Finding:** The panel was blinded, catching only 3.7% of false statements because its single alarm line was pushed up by specific layer probes, even though the layer-11 probe alone still caught 73%.

### Terms I Did Not Know
- **KL scale / nats per token**: Used in the text as a metric, but slightly unclear what the raw units mean without the foldout.
- **LoRA**: Described briefly, but still somewhat jargon-heavy for non-insiders.

### Figure Stand-Alone Check
**Result:** The figure does NOT stand alone.
- The caption lacks explicit axis definitions (what is plotted against what, e.g., the x-axis is catch rate).
- The caption refers to a 'last row added after the runs' without explicitly naming what that row represents (e.g., the rescaled panel).

### LOST
1. *"A false statement still scored above its true twin in 75% to 86% of pairs."* -> I got lost here because the previous sentences were all about the alarm line and catch rates; this sudden shift to pairwise ranking disrupts the flow.
2. *"A free route also existed. Just before the next-word prediction..."* -> I got lost here. It's unclear what this 'free route' is or who takes it. Is it a theoretical observation or a different attack? The transition is too abrupt.
3. *"The weak point is the shared scale, fixed on the original model."* -> I got lost here because it conflicts with an earlier sentence: "We set the line on the checked model's own true statements." If the line is set on the checked model, how is the scale fixed on the original model? (Wait, the scale is fixed but the line is set... this distinction is too subtle and confusing).

### READ TWICE
- READ TWICE: "The panel's one line sat at the line of the probe that scored true statements highest." -> REWRITE: "The panel's alarm line rose to match the probe with the highest true statement scores."
- READ TWICE: "The fine-tune was a LoRA, a small low-rank update of all 28 blocks." -> REWRITE: "We used LoRA to apply a small update to all 28 blocks."
- READ TWICE: "The loss pulled each false statement's probe score toward the score of its true twin, the same sentence with the other country." -> REWRITE: "The loss pulled each false statement's score toward its true twin's score."
- READ TWICE: "A probe fit again on a fine-tuned model caught 99%." -> REWRITE: "When we re-fit the probe on the fine-tuned model, it caught 99%."

## Fixes applied by Selma (blog branch swarm/theory-1022-panel-max-attack, see the commit after 0bd8c42)

| Item | New text |
|---|---|
| KL scale | "The next-word predictions change by at most 0.05 on the KL scale, a standard measure of output change where 0 means unchanged." (the foldout "What is the output-matching penalty?" defines KL) |
| LoRA | "The fine-tune used LoRA, a small update to all 28 blocks." |
| Figure, axes | The figure itself labels both axes ("Layer of the probe", "False statements caught"; bottom: "False statements caught by the panel"); the caption now opens each panel with what it shows ("Top: each probe alone at its own alarm line…", "Bottom: the whole panel at its one line…"). |
| Figure, last row | Caption: "…and, added after the runs, the panel with its shared scale set again on the fine-tuned model." Row label: "panel fine-tune, new scale*", note "*scale set on tuned model, added after the runs". |
| LOST 1 | "The panel's scores still told true from false in most pairs. A false statement outscored its true twin in 75% to 86% of pairs." |
| LOST 2 | "A free route also existed, which we built by hand on the original model. Just before the next-word prediction, the model normalises each token's activations to a fixed size. So enlarging layer 28 for 2% of true statements raised the line and cut the panel to 18%, with no output change. The fine-tune did not take this route: layer 28 shrank by only 3% to 4%." |
| LOST 3 | Intro now separates the two: "We put the 20 scores on one shared scale, set once on the original model. … We set the line on each checked model, so that 1% of its true statements cross it." Findings: "The weak point is the shared scale, which stays fixed on the original model." |
| READ TWICE 1 | "The panel's alarm line rose to match the probe with the highest scores on true statements." |
| READ TWICE 2 | "The fine-tune used LoRA, a small update to all 28 blocks." |
| READ TWICE 3 | "Each false statement has a true twin: the same sentence with the other country, like "The city of Lyon is in France." The loss pulled each false statement's score toward its twin's score, averaged over the 20 probes." |
| READ TWICE 4 | "When we fit the probe again on a fine-tuned model, it caught 99%." |

## Lumi's confirmation

(pending: Lumi confirms each fix on the refreshed preview)
