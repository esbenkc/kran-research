1. "*...the filled and hollow sets are scored separately.*"
Why: The caption and text never define what the "filled" and "hollow" sets represent (e.g., main vs. same-concept probes?). 

2. "*Filled dots bunch near zero in all three panels, and the few large drops sit far above the diagonal.*"
Why: It is impossible to know what the three panels show (e.g., the three geometries?), what the x and y axes are, or what the diagonal represents because the caption does not explain them.

3. "\\[s_u \\;\to\\; s_u - \kappa_w \\,\frac{u^\top P\\, w}{w^\top P\\, w}\\; s_w .\\] \\(\kappa_w\\) is the fraction of \\(w\\)’s class gap that the attack removed."
Why: "class gap" is an undefined term. An ML researcher would guess it means the distance between the mean activation of the positive and negative classes, but it is not formally defined.

4. "A formula with every gap change exactly right would reach 83%."
Why: "every gap change exactly right" is ambiguous. It is unclear if this refers to an oracle predictor that knows the exact shift in class means.

5. "The attack’s own first gradient step, computed without running the attack, reaches 71-79%[10]."
Why: "reaches 71-79%" is missing its metric. Context implies an $R^2$ of 0.71 to 0.79, but mixing percentages and $R^2$ decimal values is confusing.

Can you state the main claim after 3 sentences: No (the first 3 sentences are just meta-commentary about pre-registration; the actual claim comes later).
Does the figure stand alone: No (the caption mentions filled/hollow sets and three panels without defining what they or the axes represent).

---
## Selma's fixes (blog commit after 16:45, PR #33; preview rebuilds in ~1 min)

1. Caption now says what the sets are: "Filled dots are the pre-registered test, 245 pairs of a target and a probe for another concept (5 targets × 49 probes), and hollow dots are 24 probes for the target's own concept, scored separately."
2. Caption now opens: "Each panel puts one geometry into the formula: predicted AUROC drop across, measured drop up, and the dashed diagonal where the two match (both axes stretch near zero)." Note: the figure itself (title, legend, axis labels, panel names Plain angle / Whitened angle / Output Fisher) is drawn by JavaScript, so a text-only read does not show it.
3. Defined at first use: "κ_w is the fraction of w's class gap, the distance between its average scores on its two classes, that the attack removed."
4. Now: "Even the formula fed each probe's measured change in class gap explains only 83%, because the spread changed too."
5. Now "explains 71-79%", and the first R² in the results says "explains only 6% of their variance (R² = 0.06)", so percent = R² × 100 throughout.
Main claim: in the page body, sentence 3 is "On Pythia-70M, we find that such an attack barely fools probes for other concepts, and the model's own output geometry cannot say which few it does fool." The pre-registration box above the body is drawn by the layout from front matter and comes first in a text-only read.
