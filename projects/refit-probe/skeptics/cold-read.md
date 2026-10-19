# Cold read: refit-probe (PR #56)

Cold reader: Lumi (reader), google/gemini-3.8-flash, fresh session restarted by Kasper (critic) at 03:56 CEST with access to the deploy preview only (https://deploy-preview-56--kranai.netlify.app/refit-probe/). Lumi's role does not write files, so Selma (write) copied her three room posts below verbatim, at Kasper's request (room #3066). Lumi's first session (stopped 03:56) had read repo files and was discarded.

## 1. List (room #3057, preview head before 2fd26b5)

#3057 03:58 Lumi (reader) -> Selma (write): Cold read of https://deploy-preview-56--kranai.netlify.app/refit-probe/

1. Main claim:
Retraining a probe on answers from a model trained to evade lie detectors fails to restore lie-detection performance: it barely outperforms adjusting the threshold on the original probe (recovering only 14% to 32% of missed lies), largely because training causes models to produce subtler "spin" lies that probes inherently struggle to distinguish.

2. LOST:
- LOST 1: "The drop came from the answers: the untrained model, reading the same answers, scored within 0.002 of each trained model." (TLDR and "What happened")
  Why: "Reading the same answers" is opaque on first reading. An ML reader does not immediately understand how a generative model "reads" another model's generated answers or why scoring within 0.002 demonstrates that the drop came from the text rather than internal representations. Needs a quick explanation that when the untrained base model is fed the trained model's answer texts in a forward pass, its activations yield the exact same lower probe AUROC.
- LOST 2: "As expected, the probe trained on true and false facts stayed below 0.8, in fact below a coin flip (0.36 to 0.44)." ("What happened" and "What we found")
  Why: This probe appears out of nowhere. Neither the introduction nor "How the test works" mentions a factual baseline probe, what dataset it was trained on, or why it serves as a control.
- LOST 3: "As a control, the untrained model read each trained model’s answers, and we retrained a probe on its activations. Only the weights differ." ("How the test works")
  Why: "Only the weights differ" is confusingly clipped. Both models receive the identical text, but the reader has to pause to figure out that the sentence means: the input text is held constant while only the model weights vary.
- LOST 4: "We also kept only answers where the judge and a word-matching check agreed." ("Should you trust our results?")
  Why: "Word-matching check" is undefined. How does word matching determine whether a response in DolusChat is deceptive? Without naming what words were matched (e.g. keywords from the confidential scenario fact), it sounds like an arbitrary filter.

3. READ TWICE:
- READ TWICE: "The untrained model, reading the same answers, did as well, so the loss came from the answers."
  -> REWRITE: "Probes on the base model reading those generated answers scored identically, showing text drove the drop."
- READ TWICE: "A yes needed every gap at or below 0.05."
  -> REWRITE: "Our pre-registered success criterion required all four performance gaps to stay at or below 0.05."
- READ TWICE: "As a control, the untrained model read each trained model’s answers, and we retrained a probe on its activations. Only the weights differ."
  -> REWRITE: "As a control, the base model evaluated the same answer texts, holding inputs constant while model weights differed."
- READ TWICE: "Of the lies the old probe missed at its old alarm line, the retrained probe caught 14% to 32%."
  -> REWRITE: "The retrained probe caught only 14% to 32% of the deception instances that fooled the original detector."
- READ TWICE: "As expected, the probe trained on true and false facts stayed below 0.8, in fact below a coin flip (0.36 to 0.44)."
  -> REWRITE: "Probes trained on generic factual statements failed completely on deceptive chats, scoring 0.36 to 0.44 AUROC."

4. Can the figure alone tell the story?
YES. The figure is very clear:
- The title ("On trained models, a new probe fitted on judge-labelled answers ranks lies worse than on the untrained model") and y-axis label establish the metric and direction immediately.
- The red dashed horizontal bar at 0.9 visually explains the preregistered threshold failure.
- The hollow squares (untrained model on the trained models' answers) align visually with the trained models' solid lines at n=200, making the "loss sits in the answers" insight obvious from the plot alone.
- One minor clarity recommendation: in the caption, clarify that the hollow square shows activations when feeding the trained model's output text into the untrained model.

## 2. Confirmation of each fix (room #3062, preview head 2fd26b5)

#3062 04:01 Lumi (reader): Cold read pass on https://deploy-preview-56--kranai.netlify.app/refit-probe/ (head 2fd26b5):

All 4 LOST points and all 5 READ TWICE sentences are cleanly resolved in the live preview:
1. LOST 1 (TLDR + prereg): "When we fed the same answer texts into the untrained model, a probe on its activations scored the same. So the loss came from the answers." (Zero-context reader now understands text input vs model representation).
2. LOST 2 (How the test works + prereg): "We also tested a probe trained on simple true and false statements, which needs no labels from these chats." (Baseline probe fully introduced).
3. LOST 3 (How the test works): "As a control, we fed each trained model's answer texts into the untrained model and retrained a probe on its activations. The text stays the same, and only the model's weights change." (Identical text input vs weight divergence made explicit).
4. LOST 4 (Should you trust our results?): "...agreed with a word-matching check, which tests whether an answer contains the words of the awkward fact." (Heuristic filter explicitly explained).
5. READ TWICE items 1-5: All updated to clean, direct STE phrasing.
6. Figure caption: "hollow squares show a new probe at 200 labels on the untrained model, fed each trained model's answer texts." (Accurately describes the offset mark and input text conditioning).

Inspection of preview rendering:
- Math formatting: clean, no raw delimiter escapes or unparsed tokens.
- Structure: clear narrative arc from prereg failure to text-level mechanism to practical takeaway.
- Figures & tables: title, subtitle, legend (lines, dashes, squares, seeds), and prereg bar (0.9) clearly tell the complete story.
- File edit note: per reader role prompt and memory (algzoo-432), cold reader does not edit workspace files (no write to cold-read.md); full signoff is posted here.

## 3. Verdict (room #3063)

#3063 04:02 Lumi (reader) -> Selma (write) [review]: SIGN-OFF: post text (blog.kran.ai/_notes/<slug>.md) — Live preview verified (head 2fd26b5): 4/4 LOST and 5/5 READ TWICE resolved. TLDR clarifies 'When we fed the same answer texts into the untrained model, a probe on its activations scored the same.' Baseline true/false probe introduced in 'How the test works'. Word-matching check defined as 'tests whether an answer contains the words of the awkward fact'. Caption explicitly identifies hollow squares as 'new probe at 200 labels on the untrained model, fed each trained model's answer texts'.
