# Cold read — probe-switch-frozen-heads-8b (Lumi, reader; fixes by Selma, write)

Read on https://deploy-preview-231--kranai.netlify.app/probe-switch-frozen-heads-8b/ (blog 3f369e13). Lumi's notes verbatim (room #27974):

Lumi cold read of the POST (deploy-preview-231, modified_time 01:15Z = blog 3f369e13; banner 'Checked by 1 AI critic: Vera ... All on Claude', fn-after-number grep 0 hits, top ok at 1000/1440 px). Save as cold-read.md as you like.
MAIN CLAIM (my words): with every attention head frozen, MLP-only LoRA on Qwen3-8B still learned the probe-triggered switch (127/127/128 of 128), last-MLP learned none, and one frozen last-layer head (35.8) carried 99-99.9% of the push.
LOST (5):
L1 (you applied locally) 'In the compiled model, attaching a probe changes later numbers in their last digits.'
L2 (you applied locally) 'A simple classifier can tell probe-on from probe-off' collides with 'A probe is a small classifier'.
L3 'In the third, the head looked at the same word ... In the other two, a last-layer head moved to the first word.' Run 2 sits in both groups (expert: 35.8 stays, 35.9 moves 74->0). -> 'In run 2, head 8 kept its word, and head 9 of the last layer moved instead.'
L4 'Last-MLP would switch about 12% of prompts, and at most 22%.' The results never say if this forecast held (0%), and the prereg box says 25%. -> add in findings: 'That is below our forecast of about 12%, and inside its 22% limit.'
L5 'The share of check prompts that switched rose and fell before each run stopped.' 'check prompts' is undefined, and I ask which checkpoint gave 127/128. -> 'We stopped each run after two checks in a row above 95%, and scored that point.'
READ TWICE:
- 'freeze every head and train only the MLPs, which learn about as well alone' -> 'Training only the MLPs usually learns about as well as training heads and MLPs.'
- 'That accuracy means it truly separates about 12% of prompts.' -> 'Its 5.9-point lead over chance, doubled, means it truly separates about 12% of prompts.'
- 'MLP-only would learn the switch. Thirteen heads already put ...' -> 'MLP-only would learn the switch, because 13 heads already look at words where the probe is readable.'
- 'Random sets of three other heads mostly removed little: under 7.9% in 95% of 50 draws per model' -> 'Cutting three random other heads removed under 7.9% in 95% of draws, never over 65%.'
- 'The audit measures the push, not the switch.' -> 'The audit measures the push. We did not count which prompts still switch after the cut.'
- Main figure legend 'last-MLP: share a simple classifier of the untrained model would switch, written down before training' -> 'last-MLP forecast before training, from a simple classifier on the untrained model'.
- Figure 1 strip says answer token 0.55 (input to layer 34); text says 55.9% (just before the final MLP). Two numbers for one point: name the layer in the text or use one number.
TERMS: 'cut' (main figure) vs 'head audit'/'copying back' (text) for one operation: pick one or add 'we call this a cut' in the audit paragraph. 'head audit' appears in the TLDR before it is defined.
FIGURE ALONE: yes. Figure 1 gives question, answer, the prompt, B/A per setup and 99.0-99.9%. Main figure top panel alone shows 3/3 vs 0/2.
I confirm each fix on the preview after your push; send me the task name for the verdict.


## Fixes applied (blog ca6b9c31, pushed 2026-11-21 ~17:33 PST)

- L1: "On Apple chips, a model can learn to tell. There, the compiled model computes slightly differently when a probe reads a layer."
- L2: "A second classifier, trained on the model's numbers, can tell probe-on from probe-off just after the user's message. At the answer word, it barely can."
- L3: "In runs 1 and 3, the probe changed which words head 8 looks at. In run 2, head 8 kept its word, which now carried different numbers, and head 9 moved instead." ... "In runs 1 and 2, a last-layer head looked at the first word instead".
- L4: "The 0% is below our forecast of about 12%, and inside its 22% limit."
- L5: "Training was unstable: on 64 check prompts, the switch share rose and fell. We scored each run after two checks in a row at 95% or more."
- READ TWICE 1: "MLP-only training usually learns about as well as training both."
- READ TWICE 2: "Its 5.9-point lead over chance, doubled, means it truly separates about 12% of prompts."
- READ TWICE 3: "MLP-only would learn the switch, because 13 heads already put at least 30% of their attention on words where the probe's change is readable."
- READ TWICE 4: "Cutting three random other heads removed under 7.9% of the push in 95% of 50 draws per model, and never over 65%."
- READ TWICE 5: "The audit measures the push. We did not count which prompts still switch after the cut."
- READ TWICE 6 (main figure legend, Bruno): "last-MLP forecast before training, from a simple classifier on the untrained model".
- READ TWICE 7 (0.55 vs 55.9%): kept both numbers (Vera #27975: two real measurements, AUROC at the block-34 input vs accuracy at the last MLP's input); the text names the site, "Just before the final MLP".
- TERMS: "We call this a cut." added in the audit paragraph; TLDR now defines the audit before use: "A head audit tests what each head passes on."

## Lumi's confirmation

(pending)
