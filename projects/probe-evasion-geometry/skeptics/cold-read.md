Cold read of https://deploy-preview-28--kranai.netlify.app/probe-evasion-geometry, 2026-10-16 06:32 PDT (15:32 Copenhagen), reader: fresh agent with no access to the run

Read in headless Chromium, light theme, 1280 px wide, with all three folds open (the pre-registration foldout and both "If you're curious" folds), margin footnotes, and the figure. Quotes are copied verbatim from the rendered page. Math is written as MathJax renders it (e.g. "Σw"), and footnote markers as superscripts. Each item was checked against the page's text.

1. **Quote:** "Run and written by AI and rebuilt by three independent critics."
   - **Where:** Top banner (above title)
   - **Why a reader gets lost:** "Rebuilt" is unclear (re-ran the code? rewrote the post?). Later, footnote 8 names only two skeptic agents ("skeptic and skeptic2"), so the reader cannot match "three" to anything on the page.
   - **Suggested fix:** Say in plain words what the critics did (e.g. "independently re-ran the analysis") and make the count match what the footnotes name.

2. **Quote:** "It was wrong. Two parts held. The prediction from the rescaled angle explained 99.9% of the variation on synthetic data and 97% on Pythia-70M activations, and a new probe still scored 0.97 or more."
   - **Where:** Pre-registration foldout, "What happened"
   - **Why a reader gets lost:** This reads as if the 70% part held under the registered plan. But "What we found" later says that under the frozen rule (the registered stopping rule) the whitened formula explains only 51% on synthetic data, against 69% for the plain cosine. The 99.9% and 97% come from the changed protocol (16,000 steps, footnote 9). A careful reader sees two stories that disagree and cannot tell whether the headline result is the registered test or a post-hoc rerun, or when the change was decided.
   - **Suggested fix:** In "What happened", report both: under the registered stopping rule, the 70% part also failed (51%); after the one change (train until settled), it held (99.9% / 97%). Say whether the change was decided before or after seeing the 51%.

3. **Quote:** "The plain angle between the probe weights explains at most 40%."
   - **Where:** Pre-registration foldout, "What we expected" (and repeated in TLDR: "Our pre-registered prediction still failed: the plain cosine explains 47% on synthetic data, above our bar of 40%.")
   - **Why a reader gets lost:** An outsider expects the baseline doing well to be harmless. Nothing says why a ceiling on the plain angle was part of the prediction, so "failed because the baseline did too well" feels like a strange way to fail.
   - **Suggested fix:** Add one sentence where the bar is first stated: "We also predicted the plain angle would do badly, to check that it is the whitening, not any angle, that carries the prediction."

4. **Quote:** "On its own, the cosine explains 61-77% (synthetic) and 25-33% (Pythia). The formula adds each probe’s starting AUROC and k."
   - **Where:** "What we found", paragraph 1 (together with "The plain cosine explains 47% on synthetic data and 30% on Pythia." in paragraph 3)
   - **Why a reader gets lost:** There are four predictors in play (whitened cosine alone, whitened formula, plain cosine alone, formula with plain cosine), but the text names them loosely. "On its own, the cosine" does not say which cosine. "The plain cosine explains 47%" could be the plain cosine alone or the formula with the plain cosine plugged in (the figure says hollow dots are the formula with the plain cosine). The reader cannot tell which number the 40% bar was checked against.
   - **Suggested fix:** Name the four predictors once and use those names everywhere; a four-row table (synthetic / Pythia r²) would settle it. State which of them the 70% and 40% bars applied to.

5. **Quote:** "In this article, we ask can one number, the whitened cosine between two probes, predict which probes an evading model also fools?"
   - **Where:** Intro, paragraph 2
   - **Why a reader gets lost:** "Whitened cosine" is used here, in bold, two sections before it is defined. The standard name that an ML reader might recognise (Mahalanobis cosine) appears only in footnote 5.
   - **Suggested fix:** Add a short gloss at first use, e.g. "the cosine between two probe directions after rescaling activations to unit covariance (a Mahalanobis cosine; defined below)".

6. **Quote:** "Yes, for an evader that pays in the right units and has settled at its cheapest move."
   - **Where:** TLDR
   - **Why a reader gets lost:** "Pays" what, and in which "units"? "Settled" in what sense? This is the post's main condition, and an outsider cannot decode it from the TLDR alone. Only after reading two more sections do they learn that it means a cost on activation changes measured in whitened (Mahalanobis) length, and an evader trained to convergence rather than stopped early.
   - **Suggested fix:** Say it literally: "Yes, if the evader's cost for changing activations is measured in whitened (Mahalanobis) distance and it is trained until convergence rather than stopped early."

7. **Quote:** "Then a one-line formula built on the whitened cosine explains 97-99.9% of how much each of 50 probes is fooled, on synthetic data and on Pythia-70M activations."
   - **Where:** TLDR
   - **Why a reader gets lost:** The formula has not been shown yet. "How much fooled" means AUROC drop and "explains" means r² (of a fitted line, as the reader learns later), but neither is said. The title promises "a single geometric quantity", yet this formula also uses each probe's starting AUROC and k, and the cosine alone explains only 61-77% / 25-33%. The reader is left unsure whether the claim is about one number or about a formula.
   - **Suggested fix:** Rephrase, e.g. "a formula combining the whitened cosine with each probe's starting AUROC predicts each probe's AUROC drop (r² 0.97-0.999)", and say once that the cosine is the only geometric input.

8. **Quote:** "Other probes are often fooled too¹²,"
   - **Where:** Intro, paragraph 1
   - **Why a reader gets lost:** Two adjacent footnote markers render as a single superscript "12", which reads as footnote twelve (there are only ten).
   - **Suggested fix:** Separate the markers (e.g. "¹,²") or cite both in one footnote.

9. **Quote:** "On the dashed line, prediction = measurement. r² = share of the variance in measured drops that the prediction explains, mean of 3 seeds;"
   - **Where:** Figure caption (and panel subtitles "r²: whitened 0.969, plain 0.302")
   - **Why a reader gets lost:** Because r² sits next to the dashed y = x line, the reader assumes it measures how close the dots are to that diagonal. It does not: the body says that against the diagonal the Pythia score is 89-95%, so 0.969 must come from a fitted line. In the Pythia panel the filled dots are visibly off the diagonal (below it at small drops, above it around 0.3-0.4), yet r² is 0.969. This looks contradictory.
   - **Suggested fix:** In the caption, say "r² of a straight line fitted to the dots (not the diagonal)", and give the diagonal score next to it (e.g. "vs diagonal: 0.89-0.95").

10. **Quote:** "The filled dots sit on the diagonal."
   - **Where:** Text right after the figure
   - **Why a reader gets lost:** True for the synthetic panel. In the Pythia panel the filled dots sit systematically off the diagonal, which the text itself later admits ("predicts drops 0.02 AUROC too small"). The reader checks the figure, sees the offset, and loses trust.
   - **Suggested fix:** "The filled dots sit on the diagonal for synthetic data and close to it, slightly above, on Pythia."

11. **Quote:** "−0.1"
   - **Where:** Figure, both axes in both panels (dots at measured and predicted drop ≈ −0.1 in the synthetic panel)
   - **Why a reader gets lost:** A negative "drop in AUROC" is never explained. The reader cannot tell whether the probe got better after evasion, whether this is noise, or what it means for a prediction to be negative.
   - **Suggested fix:** Add to the caption: "A negative drop means the probe's AUROC rose. The formula predicts this when a probe's whitened cosine with the attacked probe is negative." Check that this is the actual reason before writing it.

12. **Quote:** "targeted probe AUROC 0.962 → 0.581, refit 0.983"
   - **Where:** Figure, synthetic panel subtitle (Pythia: "targeted probe AUROC 0.996 → 0.654, refit 0.998")
   - **Why a reader gets lost:** "Targeted probe" is not tied to the symbol w used in the text. The refit AUROC (0.983) is higher than the targeted probe's AUROC before the attack (0.962). An outsider asks how hiding a concept can leave it easier to detect than before. The likely reason (the refit is a stronger probe type than the mass-mean target) is not stated, and the figure gives no "fresh probe before evasion" baseline to compare against.
   - **Suggested fix:** Caption: "targeted probe = the mass-mean probe w the evader attacks; refit = a fresh logistic-regression probe trained on the evaded activations (the same probe type scores X before the attack)."

13. **Quote:** "Pythia-70M: layer 3, PubMed against web text from the Pile, 1,304 test windows."
   - **Where:** Figure footer
   - **Why a reader gets lost:** This is the only place that says what the real-model "concept" is (text source: PubMed vs web), while the intro's example is "a harmful request". "Windows" is not defined, so the reader cannot tell how many tokens there are, which token's activation is used, or whether activations are averaged. The probes' unit of input therefore stays unknown.
   - **Suggested fix:** In "How we tested it", add one sentence: "On Pythia-70M the concept is whether a text window comes from PubMed or from general web text; each example is an N-token window, represented by its layer-3 residual-stream activation [at the last token / averaged]."

14. **Quote:** "hollow dot: predicted from the plain cosine"
   - **Where:** Figure legend and Pythia panel
   - **Why a reader gets lost:** In the Pythia panel the hollow dots form a near-vertical column at predicted drop ≈ 0-0.1, while the measured drops reach 0.5. The reader cannot decode why the plain-cosine formula predicts almost no drop for every probe (presumably plain cosines are near zero in this anisotropic space), so the visual is striking but unexplained.
   - **Suggested fix:** One sentence in the caption or the text: why plain-cosine predictions collapse near zero on Pythia.

15. **Quote:** "There, every step of the same length changes the distribution by the same amount."
   - **Where:** "Why one angle should decide it", paragraph 2
   - **Why a reader gets lost:** "Changes the distribution by the same amount" is vague, and this is the justification for the whole cost choice. The precise statement (shifting a mean by δ costs ½·δᵀΣ⁻¹δ in KL, so equal whitened lengths give equal KL) is buried in footnote 6.
   - **Suggested fix:** Bring the KL fact into the sentence: "There, shifting a class average by any vector of length L changes the distribution by the same KL divergence, L²/2, whatever the direction."

16. **Quote:** "So if the evader pays for each change by its length in the round space, its cheapest way to hide the concept from probe w moves one class average straight along w’s direction there. In the original space, this is the direction Σw, where Σ is the covariance of the cloud."
   - **Where:** "Why one angle should decide it", paragraph 2
   - **Why a reader gets lost:** This is the key step and it is skipped. The reader must work out alone that a probe's weight vector becomes Σ^{1/2}w after whitening (because wᵀx = (Σ^{1/2}w)ᵀ(Σ^{-1/2}x)), and that mapping the move back gives Σ^{1/2}·Σ^{1/2}w = Σw. It is also unclear why only "one class average" moves rather than both, and whether "move" means shifting every point of that class by the same vector.
   - **Suggested fix:** Add the two-line derivation (as a fold or an inline aside), plus one clause: "(shift every point of one class by the same vector; moving both classes costs more for the same effect)" or whatever the true reason is.

17. **Quote:** "If the move removes a fraction k of w’s separation, every other probe u loses the part of the move that it sees³:"
   - **Where:** "Why one angle should decide it", paragraph 2
   - **Why a reader gets lost:** k is defined precisely only in footnote 3, and that footnote is broken (item 38). It is also not said that the formula assumes the spread does not change, though later results depend on this (the "widens the cloud" remarks). Without that, the reader cannot see why d′_w's loss transfers to u scaled by exactly cos_Σ.
   - **Suggested fix:** Define k in the main text ("k = 1 − d′_w after / d′_w before") and add "assuming the move leaves the spread unchanged".

18. **Quote:** "We chose this cost because the lemma assumes it."
   - **Where:** "How we tested it", paragraph 1 (first used in the figure legend: "predicted from the whitened cosine (the lemma)")
   - **Why a reader gets lost:** "The lemma" is used as a proper name throughout, but no statement on the page is labelled as the lemma. The reader has to guess that it is the displayed equation in the previous section.
   - **Suggested fix:** Label the displayed equation: "Lemma (evasion transfer): …", or replace "the lemma" with "our formula" everywhere.

19. **Quote:** "The move erases the concept along one direction only, so a fresh probe trained afterwards still finds it, unless w was already the best direction⁴."
   - **Where:** "Why one angle should decide it", paragraph 3
   - **Why a reader gets lost:** "Best direction" is undefined (best for separating the classes: the linear discriminant Σ⁻¹(μ₁−μ₀)). The reader also cannot tell whether the mass-mean w used later is that best direction. The Euclidean fold later depends on this exception ("the mirror of the exception in the lemma").
   - **Suggested fix:** "…unless w was already the best linear separator, Σ⁻¹(μ₁ − μ₀) (the direction LEACE erases). Our mass-mean w is not, unless the cloud is round."

20. **Quote:** "The evader adds a learned correction to each activation vector, f(x)=x+MLP(x)."
   - **Where:** "How we tested it", paragraph 1
   - **Why a reader gets lost:** The intro ("Train a model with a penalty each time one probe fires") and the title ("an Evading Model") suggest the language model itself is fine-tuned. Here the reader learns, only implicitly, that the evader is a separate small network that edits activations, with no language-modelling loss keeping the model working. Also unclear: does the MLP see the class label? The lemma moves "one class average", so the evader must either know or infer the class, and how strongly it is penalised is not said.
   - **Suggested fix:** State it plainly at first mention: "We do not fine-tune the language model; a small network edits its activations directly, as a stand-in for a fine-tuned model. It [does / does not] see labels." Give the penalty form, e.g. the probe's d′ on a batch.

21. **Quote:** "A Gaussian anomaly detector, one of the defences Bailey and colleagues test, uses the same metric."
   - **Where:** "How we tested it", paragraph 1
   - **Why a reader gets lost:** The reasoning is skipped: why should the reader care? The intended point seems to be that an evader that must also stay in-distribution for a Mahalanobis anomaly detector effectively pays this cost, which makes the cost realistic and not just convenient. The next sentence ("We chose this cost because the lemma assumes it") reads like the opposite justification.
   - **Suggested fix:** "This cost is realistic: an evader that must also slip past a Gaussian anomaly detector (one of the defences Bailey et al. test) pays roughly this cost. It is also the cost the lemma assumes."

22. **Quote:** "We fit 50 other probes before the evader trains: logistic-regression and mass-mean probes, each inside a random subspace so that they point in many directions."
   - **Where:** "How we tested it", paragraph 2
   - **Why a reader gets lost:** The subspace dimension is not given, so the reader cannot judge how strong or realistic these probes are. Real monitors are not limited to random subspaces. This matters because the post later blames the failed prediction on "our probes".
   - **Suggested fix:** Give the subspace dimension and the range of starting AUROCs, and say the restriction is deliberate (to spread the probes over many directions).

23. **Quote:** "Each prediction uses only the probe’s AUROC before the evader, its whitened cosine with w, and k, which we read off w."
   - **Where:** "How we tested it", paragraph 2
   - **Why a reader gets lost:** "Read off w" is unclear. The reader does not know whether k is measured from the attacked probe's AUROC after evasion (a post-attack measurement), which would be fine but must be said so readers know the prediction is not circular. The formula also needs d′_w, and the conversion AUROC → d′ → predicted AUROC (via Φ) is never stated, so the reader cannot reproduce a single predicted point.
   - **Suggested fix:** "We convert each AUROC to d′ = √2·Φ⁻¹(AUROC), apply the formula, and convert back. k = 1 − d′_w(after)/d′_w(before) is measured on the attacked probe w only; the formula never sees the other 50 probes after the attack."

24. **Quote:** "The evader takes the lemma’s move: the class average shifts along Σw (cosine −1.00 in every seed)."
   - **Where:** "What we found", paragraph 1
   - **Why a reader gets lost:** Which class average? A cosine of −1.00 means "exactly opposite to Σw", but the sentence says "along", so the reader has to work out the sign convention.
   - **Suggested fix:** "The positive-class average shifts exactly against Σw, toward the other class (cosine −1.00 in every seed)."

25. **Quote:** "Scored against the diagonal, with no fitted line, it still explains 99.6-99.9% and 89-95%."
   - **Where:** "What we found", paragraph 1
   - **Why a reader gets lost:** This is the first mention that the headline numbers (TLDR, figure, pre-registration verdict) are fitted-line r², not agreement with prediction = measurement. The reader now wants to know which of the two the pre-registered 70% / 40% bars meant. That decides whether the verdict in the foldout is right, and the page does not say.
   - **Suggested fix:** Define both scores once, early ("fitted-line r²" and "r² against the diagonal"), and state which one the pre-registration used.

26. **Quote:** "Likely causes: the classes are skewed, and the evader also widens the cloud by 15%, against under 10% on synthetic data."
   - **Where:** "What we found", paragraph 2
   - **Why a reader gets lost:** "Skewed" is ambiguous: class imbalance or non-Gaussian shape? "Widens the cloud by 15%" does not say what was measured (variance along w? total variance? std?). On synthetic data the clouds are Gaussian by construction, so the reader wonders what the "under 10%" widening is. The same issue appears in the first fold ("changes the spread of the cloud by 29-35%").
   - **Suggested fix:** Say which: "the class sizes are unequal" or "the activations are not Gaussian (skewed)"; and "the evader increases the standard deviation along w by 15%" (or whatever was measured).

27. **Quote:** "This says more about our probes than about the lemma. Their plain and whitened cosines correlate at 0.67-0.87, and in simulations run before the experiment, an evader that follows the lemma exactly also gave the plain cosine 19-65%⁸."
   - **Where:** "What we found", paragraph 3 (and footnote 8: "Posted by the skeptic agents before any evader was trained, with three probe families each: skeptic and skeptic2.")
   - **Why a reader gets lost:** The reasoning gap: if simulations run before the experiment already showed the plain cosine reaching up to 65%, why was the bar set at 40%? Were these simulations run before or after the pre-registration was frozen? Footnote 8 adds undefined terms ("skeptic agents", "three probe families"), and the reader does not know what the 19-65% range spans.
   - **Suggested fix:** Say when the simulations ran relative to the 14:52 freeze. In one clause, say who the skeptic agents are (independent AI critics who re-ran the analysis) and what a "probe family" is, and what varies across the 19-65% range.

28. **Quote:** "The lemma needs an evader that has settled. Stopped as soon as probe w reaches AUROC 0.6, as our frozen rule said, the evader moves somewhere between w and Σw."
   - **Where:** "What we found", paragraph 4
   - **Why a reader gets lost:** "Frozen rule" appears here without definition (it is the pre-registered stopping rule, explained only in the fold and in footnote 9). "Settled" is still undefined. "Moves somewhere between w and Σw" means the move's direction lies between those two directions, which an outsider has to infer.
   - **Suggested fix:** "Our pre-registered (frozen) rule stopped the evader as soon as w's AUROC reached 0.6; at that point it has not converged, and its move points in a direction between w and Σw."

29. **Quote:** "The cost decides which cosine works. With a plain Euclidean cost, the move points along w itself, the whitened formula explains at most 21% on synthetic data, and a fresh probe falls to about chance."
   - **Where:** "What we found", paragraph 5
   - **Why a reader gets lost:** "A fresh probe falls to about chance" seems to contradict the earlier rule "The move erases the concept along one direction only, so a fresh probe trained afterwards still finds it". The explanation (w is mass-mean, so a move along w closes the whole gap between the class means) appears only in the second fold.
   - **Suggested fix:** Add a clause here: "…falls to about chance, because our w is a mass-mean probe, so moving along w closes the whole gap between the class averages."

30. **Quote:** "Probes that pool over tokens with a max or with attention resist evasion better²."
   - **Where:** "Should you trust our results?", bullet 2
   - **Why a reader gets lost:** The page never says how its own probes handle tokens (one token? mean over the window?), so the reader cannot tell how far this caveat reaches.
   - **Suggested fix:** State the pooling used here (ties to item 13): "Our probes read one vector per window ([last token / mean])."

31. **Quote:** "Our prediction failed because the plain angle came along for part of the ride."
   - **Where:** Conclusion paragraph after "Should you trust our results?"
   - **Why a reader gets lost:** This idiom hides the actual mechanism, which is in the text but phrased differently: across these probes the plain cosine is correlated (0.67-0.87) with the whitened cosine, so it also predicts much of the variation.
   - **Suggested fix:** Say it literally: "Our prediction failed because, for these probes, the plain cosine tracks the whitened cosine closely (correlation 0.67-0.87), so it also predicts a lot."

32. **Quote:** "It is very clear to us that a set of monitors is only as diverse as the correlation between their scores."
   - **Where:** Conclusion
   - **Why a reader gets lost:** The phrase is logically inverted: diversity goes down as correlation goes up. It also does not say which correlation (from the earlier section, the within-class score correlation, which equals the whitened cosine). The certainty ("very clear") goes beyond what one kind of evader with one assumed cost supports.
   - **Suggested fix:** "A set of probe monitors is only as diverse as their scores are uncorrelated within a class: under this cost, probes whose scores correlate strongly fall together."

33. **Quote:** "Probes are a cheap way to watch open models. If the evader pays in round-space units, the correlations between your probes’ scores tell you, before any attack, which probes fall together."
   - **Where:** Conclusion
   - **Why a reader gets lost:** "Open models" (why only open-weight ones?) and "round-space units" are informal jargon. A practitioner cannot tell what to compute: correlation on which examples (one class), with which probe (the one the attacker targets), plus the starting AUROC and k that the formula also needs.
   - **Suggested fix:** "…the within-class correlation of each probe's score with the attacked probe's score, together with its starting AUROC, predicts how much it will drop." Drop "open" or say why it matters.

34. **Quote:** "On Pythia, the whitened formula still ranks the probes (77-88%, plain cosine 25-30%), but with no fitted line it explains only 49-57%, and the evader changes the spread of the cloud by 29-35%."
   - **Where:** If you’re curious, fold "What if the evader stops as soon as the probe is fooled?"
   - **Why a reader gets lost:** "Ranks the probes (77-88%)" mixes a ranking claim with what is presumably a fitted-line r², not a rank correlation. The reader cannot tell which metric the percentage is.
   - **Suggested fix:** "With a fitted line it explains 77-88% (plain cosine 25-30%); against the diagonal only 49-57%." If ranking is meant, report Spearman ρ.

35. **Quote:** "a formula built for the Euclidean move explains 99.6-99.9%¹⁰"
   - **Where:** If you’re curious, fold "What if the evader pays a plain Euclidean cost?"; footnote 10 in the right margin
   - **Why a reader gets lost:** BROKEN RENDERING: the footnote 10 formula is cut off at the right edge of the margin note. It shows "d′u → d′u − k d′w cos(u, w) ‖u‖√ w⊤Σw / (‖w‖√u" and then stops, so the closing "⊤Σu)" is hidden. The square-root bars also render detached from their arguments. The reader cannot read the formula.
   - **Suggested fix:** Let margin-note math wrap or scroll horizontally, or move this formula into the fold body as a displayed equation.

36. **Quote:** "On Pythia (two seeds), the whitened formula explains 1-7% and the Euclidean one 70-72%."
   - **Where:** If you’re curious, fold "What if the evader pays a plain Euclidean cost?"
   - **Why a reader gets lost:** The main runs use three seeds and nothing says why this run has two. The Euclidean formula reaches 99.6-99.9% on synthetic data but only 70-72% on Pythia, and the gap is not commented on.
   - **Suggested fix:** Add a clause on why there are two seeds, and a short likely reason for the 70% (e.g. the same skew / spread change noted for the main Pythia run).

37. **Quote:** "This is the mirror of the exception in the lemma: whichever cost the evader pays, attacking the probe that is best in that cost’s geometry erases the concept for every linear probe."
   - **Where:** If you’re curious, fold "What if the evader pays a plain Euclidean cost?"
   - **Why a reader gets lost:** The reader must rebuild several links: "the exception" = "unless w was already the best direction" (two sections earlier); under a Euclidean cost the "best" probe is the mass-mean direction; under the whitened cost it is Σ⁻¹(μ₁−μ₀). Without these spelled out the sentence is a riddle.
   - **Suggested fix:** Spell out both cases: "With the whitened cost, attacking the best separator Σ⁻¹Δμ moves the mean along Δμ and erases the concept. With the Euclidean cost, attacking the mass-mean probe Δμ does the same. Either way, attacking the probe that is optimal for the evader's own cost removes all linear information."

38. **Quote:** "The fraction k is 1 − d′w(extafter)/d′w(extbefore)."
   - **Where:** Footnote 3 (right margin, beside the figure / "Why one angle")
   - **Why a reader gets lost:** BROKEN RENDERING: \text lost its backslash (it became a tab), so the formula renders as "d′w(extafter)/d′w(extbefore)". This is also the only place k is defined. Separately, the "theory/lemma.md" link goes to the theory/ folder, not to the file.
   - **Suggested fix:** Escape \text properly so it renders "1 − d′_w(after)/d′_w(before)", and point the link at theory/lemma.md itself.

39. **Quote:** "For an evader trained to its cheapest move, one angle predicts which probes it also fools"
   - **Where:** Figure title (second pass, page at commit 1bc03c9)
   - **Why a reader gets lost:** A reader who looks only at the figure sees a clean success. Nothing in the title, caption or footer says this is the follow-up run, chosen after the pre-registered result was seen. The pre-registered early-stop run, where the whitened formula lost (r² 0.51 vs 0.69), has no figure. The TLDR says this, but the figure does not.
   - **Suggested fix:** Add "(follow-up run, not the pre-registered one)" to the caption. Optionally add a small early-stop panel so the failure can be seen too.

40. **Quote:** "The evader is a small network that edits activations directly, f(x) = x + MLP(x), without seeing labels."
   - **Where:** "How we tested it", paragraph 1 (second pass)
   - **Why a reader gets lost:** The lemma describes shrinking the gap between the class averages while the spread stays the same. A label-blind edit can only act on where a point lies. The obvious way for it to close a gap along a direction also squeezes the spread along that direction. The reader cannot see how the evader produces the lemma's pure mean shift without labels, or how this relates to the measured covariance changes (15% on Pythia, under 10% synthetic).
   - **Suggested fix:** Add one sentence on how a label-blind edit shrinks the class gap (e.g. it infers the class from x, since the classes are separable). Say that what it changes beyond a pure shift is what the covariance-change numbers measure.

41. **Quote:** "Our lemma is their gradient-alignment argument, with the attack’s budget measured in the round space."
   - **Where:** Footnote 4 (second pass), next to the body's "We found no prior use of it for evasion."
   - **Why a reader gets lost:** The footnote calls the lemma an existing argument, and the body says the use for evasion is new. The two read as conflicting on what is new.
   - **Suggested fix:** State the split in one line: "The argument is Demontis et al.'s; what is new is applying it to probe evasion with a whitened budget and testing it."

## Overall

Partly yes. After the TLDR and the figure, an outside ML researcher would say: "When a network is trained to hide a concept from one linear probe, how much each other probe's AUROC drops is predicted almost perfectly by a formula based on the covariance-weighted (whitened) cosine between that probe and the attacked one, much better than by the plain cosine." That matches the post. They could not state the conditions the claim depends on: "pays in the right units" and "settled at its cheapest move" are opaque until two sections later (items 6, 28). They would read the figure's r² as agreement with the dashed diagonal, which it is not (item 9). They would be confused that the answer is "Yes" while the pre-registered prediction "failed" because a baseline did too well (item 3). Read to the end, they would also find that under the registered stopping rule the headline formula explained only 51% and lost to the plain cosine, which the foldout's "Two parts held" does not reflect (item 2). That is the most damaging gap for trust. Two rendering bugs, both in margin footnotes, block reading: footnote 3 (\text lost, item 38) and footnote 10 (formula clipped, item 35). Not counted as an item: the Netlify "Collaborate on this Deploy Preview" bar covers the Pythia panel's x-axis while scrolling, but only on the deploy preview.

## Fix confirmations

Second pass: 2026-10-16, deploy preview at commit 1bc03c9 (hard reload, all folds open, read in headless Chromium at 1280 px and 1600 px). Verdicts: 31 fixed, 4 partly fixed, 1 not fixed, 2 no longer apply; 3 new items (39-41).

- [x] item 1: fixed. Footnote 11 now says the three skeptic agents are "independent AI critics that re-derive the math and re-run the analysis", and two of them ran the simulations.
- [x] item 2: fixed. "What happened" now says "It failed under both stopping rules": 51% under the pre-registered rule, then 99.9%/97% in a follow-up "chosen after we saw that result".
- [x] item 3: fixed. The body now says the bar "checks that the rescaling, and no plain angle, carries the prediction".
- [x] item 4: fixed. The page now uses "whitened formula", "plain formula" and "whitened cosine alone" consistently, and the pre-registration says "The same prediction with the plain angle".
- [x] item 5: fixed. The intro defines the whitened cosine before using it.
- [ ] item 6: partly fixed. The TLDR now says "converged evader" instead of "settled", but it never states the cost condition (whitened cost), even though the conclusion makes the cost the main takeaway. "Does not follow the theory" refers to a theory the TLDR has not introduced.
- [x] item 7: fixed. The TLDR names the metric (r²) and the inputs ("with each probe's starting AUROC, the whitened cosine predicts…").
- [x] item 8: fixed. The two references are now one footnote (1).
- [x] item 9: fixed. The caption defines r² ("credit for any straight line") and "R² on the line" separately, and each panel shows both.
- [ ] item 10: partly fixed. The text says "Filled dots sit on the diagonal (just above it on Pythia)", but in the Pythia panel the filled dots sit below the line for predicted drops under about 0.15 and above it beyond. They cross the line rather than sitting just above it.
- [x] item 11: fixed. The caption says "A negative drop means the probe got better."
- [ ] item 12: partly fixed. The panel now says "probe w: AUROC 0.962 → 0.581" (tied to w), but "new probe trained after the attack: AUROC 0.983" is still higher than w's AUROC before the attack. The page never says what type of probe the new probe is.
- [x] item 13: fixed. Footnote 9 gives the PubMed vs web-text concept, 128-token windows, and the layer-3 residual stream averaged over each window.
- [ ] item 14: not fixed. The Pythia hollow dots still form a column near zero predicted drop with no explanation, and the text says "Hollow dots scatter", which they do not on Pythia.
- [x] item 15: fixed. The body now says "costs the same KL divergence, L²/2, in every direction".
- [x] item 16: fixed. Footnote 2 derives Σw, and the body now talks about the gap between the class averages instead of "one class average".
- [x] item 17: fixed. The body defines k and adds "If the spread does not change".
- [x] item 18: fixed. The displayed equation is introduced as "the lemma we test".
- [x] item 19: fixed. "best linear separator" is used, and footnote 3 gives Σ⁻¹(μ₁ − μ₀) and says the mass-mean w differs from it unless the cloud is round.
- [x] item 20: fixed. "We do not fine-tune a language model", "without seeing labels", and footnote 6 gives the loss form (new question about this in item 40).
- [x] item 21: fixed. The cost is now justified together with the Bailey anomaly detector in one sentence.
- [x] item 22: fixed. Footnote 10 gives the subspace dimensions (2 to the full dimension), the probe mix and the range of starting AUROCs.
- [x] item 23: fixed. The page states the AUROC → d′ → lemma → AUROC pipeline, and k is "measured on w alone".
- [x] item 24: fixed. The page now says "the gap … shrinks exactly along Σw (cosine 1.00 in every seed)".
- [x] item 25: fixed. "the r² of a straight line … as pre-registered, and against the diagonal".
- [x] item 26: fixed. Footnote 12 says "the web-text scores are skewed" and gives the covariance change as a relative Frobenius norm.
- [ ] item 27: partly fixed. Footnote 11 now explains the skeptic agents and the six probe sets. It still does not say whether the simulations came before the 14:52 pre-registration freeze, or why the bar was 0.40 when they already showed plain r² up to 0.65.
- [x] item 28: fixed. The page uses "pre-registered rule" and "converged" and defines both in place.
- [x] item 29: fixed. "moves along our mass-mean w itself, which closes the whole gap between the class averages".
- [x] item 30: fixed. "reading one averaged vector per window".
- [x] item 31: fixed. The idiom is replaced with "the plain cosine tracks the whitened one closely (correlation 0.67-0.87)".
- [x] item 32: no longer applies. The sentence was replaced by "every evasion audit must name the cost the evader pays".
- [x] item 33: no longer applies. The "open models" and "round-space units" sentence was removed.
- [x] item 34: fixed. The fold now says "0.77-0.88 with a fitted line … but only 0.49-0.57 against the diagonal".
- [x] item 35: fixed. The Euclidean formula is now a displayed equation in the fold body and renders in full (the square-root bars look slightly detached, which is cosmetic only).
- [x] item 36: fixed. "two seeds" was removed, and the Pythia gap is explained ("likely because its activations are not Gaussian").
- [x] item 37: fixed. The mirror argument is spelled out for both costs, with μ₁ and μ₀ defined.
- [x] item 38: fixed. k is defined in the body and renders "after"/"before" correctly, and the theory/lemma.md link points at the file.
- [ ] item 39: new (non-blocking)
- [ ] item 40: new (non-blocking)
- [ ] item 41: new (non-blocking)

Sign-off: YES
