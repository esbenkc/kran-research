# Cold read: "What happens if you push each neuron up?"

Source: deploy-preview-29, fold under "If you're curious". Reader: an ML researcher who does not work in mech interp. Checked in the browser, light and dark mode, both toggle states.

## 1. Sentences where the reader gets lost

1. Fold title: "What happens if you push each neuron up?"
   Fix: "each" is wrong, since only a sixth of the neurons (a twelfth at 10×) were pushed. Say "a neuron".

2. "We added this answer after the post went up."
   Fix: also say whether the four predictions further down were written down and time-stamped before the run, like the main prediction was. As it stands the reader can't tell how much weight to give them.

3. Figure subtitle: "Share of runs a blind judge labels one theme, several unrelated themes, or no visible effect."
   Fix: "run" is first defined in the caption under the figure. Define it here: "a run = one pushed neuron or direction, three 20-token continuations". Also say what the judge is blind to.

4. Figure subtitle: "96 random directions per layer at the same strength"
   Fix: say which strength. Each direction takes the max of one sampled neuron, but the reader only learns that much later, from "borrow a top-fifth strength".

5. "We added each neuron's largest value on 200,000 Pile tokens to that neuron at every token."
   Fix: say what was added to (the activation after the nonlinearity?) and where (prompt tokens, generated tokens, or both). Also say why the max comes from 200k tokens when the audit above used 5M. Otherwise the reader wonders whether it is the same "max".

6. "The controls push a random direction in the same layer by the same amount, or push nothing."
   Fix: this is the key control, and it is underspecified. Say: a random unit vector in the 3,072-neuron MLP space (or in the residual stream?), scaled by one sampled neuron's max. Then say whether the change it makes to the residual stream has the same size as the change a neuron push makes. If the sizes differ, "pushed as hard" is not a fair comparison.

7. "Then the model continued 'The following is a short passage of text.' three times, for 20 tokens each."
   Fix: give the sampling setup (temperature, same seeds with and without the push?). Without it, the reader can't tell how different two unpushed texts can be by chance.

8. "The judge sees nine typical continuations and the three new ones, never the arm."
   Fix: say that "typical" means continuations from the unpushed model. Define "arm" once (neuron / random direction / no push) or drop the word.

9. "It labels the three as typical, one shared thing (mono), or several unrelated things (poly)."
   Fix: use one word. The figure says "theme" and the text says "thing". Say what counts as a "thing" (a topic, a token, a style?).

10. "A second call checks each 'one thing' answer: do at least two texts share the same thing?"
    Fix: say what happens when the answer is no: does the run become "several" or "typical"?

11. "The no-push rate measures the judge's false alarms, about half of the 11%."
    Fix: do the subtraction for the reader: "after removing false alarms, about 5% of neurons and 3% of random directions show a real effect."

12. "The neurons' small lead comes from the fifth with the largest max values (18% against 10%)."
    Fix: say what 18% is compared with: top-fifth neurons against random directions given the same strengths, or against the other neurons? Also, the heatmap uses tenths while the text uses fifths. Use one.

13. "This close to the noise, we do not split the effects into one thing or several."
    Fix: the figure does split them at max (pink and teal segments, and teal is most of each bar). Either show the max bars as one "any effect" colour or change the sentence.

14. "Mostly nothing happens."
    Fix: this is only true at max. At 10×, 68% of runs change. Say "At a neuron's own max, mostly nothing happens."

15. "The one thing is often a single token in a loop ('with with with', '###')."
    Fix: draw the conclusion: "so at 10×, 'one theme' mostly means the model has broken into repetition, not that it shows a concept." Right now the reader has to work out that the 55% "mono" is mostly breakage.

16. "In 46% of the neurons' one-thing runs, at least two of the three texts compress better than 99% of no-push texts."
    Fix: say that "compress better" means more repetitive (name the compressor). Give the same number for random directions, or the reader can't tell whether 46% is high.

17. "A check without any judge gives the same picture: a rare word in at least two of the three texts marks none of the 192 no-push runs, ..."
    Fix: define the check: "the same word, one that appears in under X% of no-push texts, shows up in at least two of the three continuations."

18. "So the small gap at 10 times flips with the measure."
    Fix: spell it out: "the judge puts random directions 3 points ahead (71% to 68%), the rare-word check puts neurons 4 points ahead (27% to 23%), so neither gap means anything."

19. "...on held-out runs only 5 of the judge's 12 'several' calls matched our hand label."
    Fix: say what the runs were held out from (the 100 runs not used to tune the prompt?).

20. "...and they called more than half of the ordinary texts changed."
    Fix: say that "ordinary" means no-push runs, and use one word for them throughout ("typical", "ordinary", "no-push" are all used).

21. "One thing beats several among the effects: held at 10 times, and at max we cannot tell, because the split moves with the judge's second call."
    Fix: plain version: "at max, the mono/poly split changes depending on whether we apply the second check, so we don't report it."

22. "Neurons beat random directions: held at max by 2 points, failed at 10 times."
    Fix: give a CI or test for the 2 points. The fold has already called this region "close to the noise", so a bare "held" reads as overclaiming.

23. "Effects gather in the largest max values: held, for random directions as well."
    Fix: random directions have no max of their own. Say "random directions given a large neuron's strength also change the text more."

24. "Push size decides."
    Fix: say what "size" is: the scalar added to the activation, or the size of the change in the residual stream. This ties back to item 6.

25. "So this simple form of steering cannot count one-job neurons, and it does not test Cammarata's 90%."
    Fix: say which reading you mean and why you rule out the other. Reading A: the method can't tell units apart. Reading B: on the output side, neurons are no more special than random directions. A reader can take either from the data.

26. "Code, raw texts, labels and the skeptics' rebuilds: projects/neuron-sweep."
    Fix: the link text is a bare repo path pointing to a working branch. Use a normal link label, e.g. "code and data", that points to a stable location.

## 2. Figure

### Both modes
- The toggle buttons have no `aria-pressed` or other state attribute. The only cue for the active state is which button is filled.
- Switching the site theme (the real `#theme-toggle` and the `themechange` event) resets the figure to "Pushed to its max activation", even when 10× was selected. Someone who switches theme to read the 10× view loses it.
- At max, the bars split effects into mono (pink) and poly (teal), which the text says it does not do at max (item 13).
- The "random directions" control label is rotated and sits on top of its own bar. At 10× it covers the pink and teal segments, and both the label and the bar are hard to read. "no push" sits in a separate narrow column with no tick.
- The control bars get no hollow dot and no tick label. The "no push" bar looks the same in both toggle states (it is the same 192 runs), but nothing says so, so it looks like a 10× control.
- The tooltip shows only the hovered segment ("One theme (mono): 28.9%"). To get the total effect for a layer, the reader has to add segments by eye.
- Heatmap: the intro line ("Same neurons, by layer and by the size of their max activation ...") does not say it follows the toggle, or that it shows neurons only, not random directions.
- Heatmap y axis: only "largest" and "smallest", with no axis title. The tooltip says "tenth 6 of 10", where 10 is the largest, which is the reverse of the usual "top tenth = 1".
- The heatmap uses tenths, the text uses fifths ("top fifth"). Cells hold 20–35 neurons (from the tooltips), so single cells are noisy. Nothing says so, and the bright or dark single cells look meaningful.
- Axis and tick text is small. The SVG is 700 px wide but drawn at about 558 px, so 10–13 px labels show at about 8–10 px.
- The heatmap colour-legend label "0%" sticks out about 8 px left of the figure box, and gets cut off when the figure is viewed or screenshotted on its own.

### Light mode
- Active toggle: solid black fill, readable.
- The low end of the heatmap scale is near-white on a white page. In the max state the whole heatmap looks blank or washed out, and cells near 0% look like empty space.
- The "No visible effect" segments (light beige) are fine against white.

### Dark mode
- The "No visible effect" colour (#282726) on the page background (#100f0f) has a contrast of about 1.3:1. The legend swatch is close to invisible and the bars' 100% outline almost disappears.
- The low end of the heatmap scale (about rgb(24,18,21)) is almost the page background. In the max state the heatmap reads as a black rectangle with a few dim cells, and the near-0% cells (e.g. around layer 18) look like holes.
- Active toggle: light fill, the opposite of light mode. That is fine, but with no other state cue a first-time reader may take the outlined button as the active one.

## 3. Could I state the result?

Adding a neuron's own max activation to it rarely changes Qwen3-0.6B's output (11% of runs, against 9% for a random direction of matched size and a 6% judge false-alarm rate). At 10× most pushes change the output, but neurons and random directions do so equally often and in the same way (mostly repetition loops), and the size of the push, not whether it is a neuron, predicts the effect. So this kind of push-up steering can't tell one-job neurons from random directions, and says nothing for or against Cammarata's 90%.

## Confirm pass

Re-read the hard-reloaded preview after the controls-panel fix. Checked light and dark mode in both toggle states, plus tooltips on the layer bars, control bars and heatmap.

### Text items
1. FIXED. The title is now "What happens if you push a neuron up?".
2. FIXED. The fold now says the predictions "went into our time-stamped log before the judge labelled anything". Minor: the log is not linked.
3. FIXED. The subtitle defines a run and says the judge is blind to the push.
4. FIXED. "Each random direction gets the max of a random neuron in its layer."
5. FIXED. It now says: after the nonlinearity, at every token, prompt and new text, and why 200k tokens.
6. FIXED. The random direction is a mix of the 3,072 neurons scaled to one neuron's max, and the change to the internal state has matched size (median ratio 0.95–1.10).
7. FIXED (temperature 0.8 given). Minor: the fold still does not say whether pushed and unpushed runs use the same seeds.
8. FIXED. "typical" is now defined as no-push text, and "arm" is replaced by "kind of push".
9. FIXED. "thing" is defined (word, topic, script, symbol or format), and the figure legend now says "One shared thing / Several unrelated things".
10. FIXED. "If not, the run counts as several."
11. FIXED. "Minus those, about 5% of neurons and 3% of random directions..."
12. FIXED. The comparison is now "against 10% of random directions with the same strengths", and the heatmap uses fifths.
13. FIXED. At max the figure shows one unsplit "Visible effect" colour, and the text says the split is not reported.
14. FIXED. The fold now opens with "At a neuron's own max, mostly nothing happens."
15. FIXED. "So at this strength, 'one thing' often means the model broke into repetition."
16. FIXED. zlib is named and the random-direction figure (41%) is given.
17. FIXED. The rare-word rule is defined.
18. FIXED. "the judge puts random directions 2.5 points ahead. So neither gap at 10 times means much." Minor: the neurons' 4-point lead on the rare-word check is left for the reader to work out.
19. FIXED. "On 100 runs held out from writing the prompt".
20. FIXED. "typical" is used throughout.
21. FIXED. "at max we do not report the split".
22. FIXED. "p = 0.001 in a permutation test".
23. FIXED. "random directions given those strengths show the same rise".
24. FIXED. Push size is now tied to the matched internal-state change in item 6.
25. FIXED. "does not single out neurons ... says nothing for or against Cammarata's 90%, which counts what neurons read."
26. FIXED (the link label is now "Code and data"). Minor: it still points to a working branch (`tree/swarm/oracle-1015-already-clean/...`), which may break after merge.

### Figure items
- Toggle state: FIXED (`aria-pressed` true/false).
- Theme switch resetting the toggle: FIXED. The 10× state survives the site toggle in both directions.
- Max split into mono/poly: FIXED.
- "random directions" label sitting on its own bar: FIXED. The controls now sit in their own panel to the right with labels underneath (not clipped, after the latest deploy).
- Controls had no ticks, and no-push looked like a 10× control: FIXED. Labels sit under the bars, % labels are on the bars, and the tooltip says "the same 192 runs at both strengths".
- Tooltip showed only one segment: FIXED. The 10× tooltip now also gives "any visible effect".
- Heatmap did not say it follows the toggle or shows neurons only: FIXED. "Neurons only, at the strength chosen above".
- Heatmap y axis: FIXED. It now has the axis title "Max activation" and names each fifth from largest to smallest.
- Tenths vs fifths, and n per cell: FIXED. Fifths are used, and the cell count is stated and updates with the toggle (about 100 at max, about 50 at 10×; tooltips show 96–106 at max).
- Small text: FIXED. The SVGs are no longer scaled down (10–14 px).
- "0%" legend overflow: FIXED.
- Light: heatmap low end blending with the page: FIXED. The low end is now beige and the cells stay visible at max.
- Dark: "No visible effect" contrast: FIXED (#575653 on #100f0f).
- Dark: heatmap low end close to the page: FIXED (about rgb(48,42,44) on rgb(16,15,15); cells visible at max).
- Dark: active toggle cue: FIXED via `aria-pressed`. The visual fill still flips between modes, which is acceptable.

### New places a reader gets lost (none blocking)
1. The control-panel labels run into each other. The "random directions" text spans x 416–481 and "no push" spans 478–531 on the same line, so it reads "random no push / directions". The label is also one text node, "randomdirections" with no space, which screen readers read as one word. Fix: put each label centred under its own bar, or wrap "no push" too.
2. The y-axis label "100%" on the layer chart sticks out 4 px left of the figure box, and the "1" is cut off when the figure is viewed or screenshotted on its own.
3. "in under 0.5% of all texts at max": it is unclear whether rarity is measured on pushed texts at max and then reused at 10×. Fix: say which pool of texts sets "rare".
4. The subtitle writes "10x max" while the button and text use "10×". This is cosmetic.
5. The control-bar tooltips dropped the 95% CI that the layer-bar tooltips show.

VERDICT: CONFIRMED
