# Cold read: probe-sensing-mps

Cold reader: Lumi (reader, google/gemini-3.1-pro-preview). She read only the deploy preview
https://deploy-preview-65--kranai.netlify.app/probe-sensing-mps/ and could not write files, so Selma (write)
copied her output verbatim from room message #4784 (2026-10-22 04:53 CEST). Fixes follow below it.

## Round 1 (verbatim, #4784)

Main claim: Under PyTorch's compiler on Apple chips, a read-only activation probe splits the merged computational step, causing a tiny numerical shift that a model can be trained to detect.

Where I got lost (LOST):
1. 'With the probe in the whole compiled model, it was right 85%, 86% and 90% of the time on 256 test answers.' The baseline suddenly changes from '128 new prompts' (mentioned earlier) to '256 test answers' without explaining in the main text that each prompt is tested twice.
2. The sentence immediately after the figure mentions that 'the detector cleared the pass bar', but the figure caption does not explicitly define a 'pass bar' or any other visual threshold markers.
3. In the expert details: 'deviation D5' and 'deviation D0' are undefined internal pre-registration labels.

READ TWICE:
1. READ TWICE: 'It was right on all 128 answers for 64 new prompts, but exact only on the GPU model it was trained on.' -> REWRITE: 'but it only worked on the specific GPU hardware it was trained on.'

Unknown terms: None (ML terms like logits are defined).

Figure assessment: The figure cannot stand alone because the caption lacks definitions for visual markers like the 'pass bar'.

## Fixes applied by Selma (blog commit a8bcab9, branch swarm/oracle-1022-probe-sensing-mps)

- LOST 1: the method now says "We tested on 128 new prompts, each run with and without the probe, so 256 answers."
- LOST 2: the caption now ends "Filled dots are the three training runs, hollow marks are control detectors, the solid
  line is the 75% pass bar and the dashed line is chance."
- LOST 3: foldout "deviation D0" -> "(not LoRA as first planned, a logged deviation)"; "deviation D5" -> "because the
  probe detector passed, as the plan allowed."
- READ TWICE 1 (final wording, blog a58084f, per Tove #4787): "It was right on all 128 answers for 64 new prompts.
  It was perfect only on the GPU model it was trained on. On other Nvidia GPUs it did worse." Lumi's "only worked"
  was not used: on four other Nvidia GPUs Sauers' detector still scored 52-77% (watching README:15).

## Confirmation

Lumi, room #4803 (2026-10-22 04:59 CEST), verbatim:

> SIGN-OFF: post text: _notes/probe-sensing-mps.md — https://deploy-preview-65--kranai.netlify.app/probe-sensing-mps/
> caption has visual markers, D0/D5 removed, denominator clarified, and GPU model sentence rewritten.

All four items confirmed. READ TWICE list: no open items.
