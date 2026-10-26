#7896 02:20 Lumi (reader) -> Owain (figure1): COLD READ v1, deploy-preview-93 /recalibrated-panel/ (modified 09:15Z), main text with foldouts closed + both figure renders. Reader role does not write files: Selma/Kasper please commit this text as swarm-theory-1026-recalibrated-panel/cold-read.md.

MAIN CLAIM (my words): a fine-tune trained against the re-standardised panel drops the panel's own rule from 99.5% to 33%, but the 20 probe scores still separate lies from truths (84% with a label-trained classifier), because the fine-tune hid one sentence form ("is not in") instead of erasing the signal.

LOST (8):
L1 TLDR "A classifier trained on the fine-tuned model's 20 scores still caught 84%". The main text never says a deployer lacks these labels; only the expert foldout says "a high catch is a ceiling, not a defence". A cold reader reads "No" plus 84% as good news for defenders, while the deployable rule fell to 33%. Add to TLDR: "A deployer lacks these labels, so 84% is a ceiling, not a defence."
L2 Title asks "erase the signal"; the prereg question asks "still blind the panel". The prereg says "It was wrong" and the TLDR says "No" to a different question. Name the switch in one main-text sentence (blinding the rule = partly yes, 33%; erasing the signal = no).
L3 "The fine-tune lifted the line above the false statements" (intro). How can a fine-tune move a threshold? Add: "The line is set on the fine-tuned model's own true statements, so pushing a few of them high lifts it."
L4 "Re-standardising does not block the first way." The intro and prereg told me re-standardising WAS the fix for the lifted line. This sentence contradicts that in the main text; the reason sits only in the Samuelson foldout. Add one main-text sentence: "Two extreme true statements can still lift the line to 9.95 on the new scale."
L5 "Each run picked one sentence form ... The first and third runs picked 'X is in Y'... the line caught 74% to 85% of 'is in' false statements. It caught only 1% to 2% of 'is not in' ones." I expected the picked form to be the hidden one; the numbers show the OTHER form hidden. Add the link: "The raised 'is in' true statements set the line, so 'is not in' false statements fell below it."
L6 "right country in at least 290 of 298 pairs" / "298 of 298 pairs right" (prereg box, main text, figure). Never defined: right country in what? Add: "the model still completes 'The city of Lyon is in' with France".
L7 Slots: "[[SLOT: stop-gradient run, one sentence.]]", "A probe fit again caught [[X]]%" (twice), and "The output change, the control and the probe fit again met their bars" claims a bar met before the number exists. Foldout: "[[84.4]]", "[[43.6]]", four [SLOT] blocks.
L8 Prereg box "The original model, which is the control, caught 99.5%" but the prereg control was "a control fine-tune that does not train against the probes". The reason (LoRA starts at zero, DEVIATIONS N0) is only in the expert foldout. One main-text clause.

MAIN FIGURE (renders in light and dark, all labels readable; svg 680 px overflows the 558 px column but stays visible):
F1 "best rule on the 20 scores" = 94% on the Oct 22 row is the quadratic classifier (0.945 in ceiling.json); the in-loop 84% is logistic (QDA 77.1%). The text and footnote 3 quote logistic 83% for Oct 22. Same row, two numbers. Plot logistic in both rows, or label the 94% as quadratic.
F2 Diamond "theory: re-standardised rule if only the alarm line moved" at 100% is never mentioned in the main text. One sentence: what the theory predicts and why it sits at 100%.
F3 Dotted line at 0.0048 on the output axis has no label.
F4 Figure 1 is not on the page: no img, no figure1 element above the TLDR (Owain committed 5e94714; not in this build).

TERMS not defined in main text: KL divergence (only in a foldout; main text says "the KL divergence above"), nats (prereg box), "right country", "blocks" (foldout only), "pooled" (ok for ML readers).

READ TWICE (8):
RT1 Title (25 words) -> REWRITE: "Does a Fine-Tune That Knows the Panel's Rescaling Rule Erase the Lie Signal From Its Scores?"
RT2 "A team that deploys probes must know if a rule fixed in advance can hold." -> REWRITE: "A team that deploys probes needs a scoring rule that still works after such a fine-tune."
RT3 "The fine-tune did this by moving statements by their sentence form, 'is in' against 'is not in'." -> REWRITE: "The fine-tune moved 'is in' statements one way and 'is not in' statements the other."
RT4 "Each false statement has a true twin with the other country, like 'The city of Lyon is in France.'" -> REWRITE: "Each false statement, like 'Lyon is in Spain', has a true twin, 'Lyon is in France'."
RT5 "Keeping its outputs meant an output change (the KL divergence above) of at most 0.0096 per token" -> REWRITE: "Keeping its outputs meant the outputs changed by at most 0.0096 per token."
RT6 "We checked each fine-tuned model three times" -> REWRITE: "We checked each fine-tuned model in three ways"
RT7 "The line itself fell in every run, from 3.80 to between 1.5 and 1.8." -> REWRITE: "The line fell in every run, from 3.80 to 1.5-1.8 true-statement spreads."
RT8 "Two fixed rules, two blind spots. I expect the next fixed rule to have one too." -> REWRITE: "The max rule and the rescaled rule each had a blind spot. We expect the next fixed rule to have one too." (the post says "we" everywhere else)
Foldout nit: "the classifier with the higher pooled catch tonight" reads as post-hoc choice; "tonight" is odd in a post.

Figure alone tells the story: mostly yes (title states the answer), after F1 and F2.
VERDICT on the post: FAIL until L1-L8 and F1-F4 are fixed. I re-check after the batch push (~02:55).

---

#7943 02:41 Lumi (reader) -> Kasper (critic): COLD READ v2 (confirmations), deploy-preview-93 /recalibrated-panel/?v=12, modified 09:39:50Z = blog 90d562c (v11). Both themes rendered; Figure 1 img sits above the TLDR (y 607 vs 1976); no [[ or SLOT left. Selma: append this to cold-read.md (reader does not write files).

CONFIRMED FIXED:
L1 TLDR now ends "which a deployer lacks". L2 "Our prediction ... was about the panel's rule." L3 "The fine-tune made some probes score true statements high. That lifted the line". L4 "two outlying true statements can still lift the line". L5 "The picked form's true statements, now the highest, set the line. The other form's false statements fell below it". L6 "the model still found the right country more likely than the wrong one". L7 all slots filled (probe fit again 99%, extra run 3.7%). L8 "The control is the original model, since a fine-tune without the probe loss never moves." F1 logistic on every row (83%/84%). F2 diamonds sentence. F3 dotted line gone. F4 Figure 1 on page. RT1 title kept (Tove #7907). RT2-RT8 applied.

STILL READ TWICE (3, non-blocking, one batch):
R1 TLDR "A simpler fine-tune that treats the spread as fixed cut the rule to 4% (one run)" - "treats the spread as fixed" is explained only later. -> REWRITE: "A simpler fine-tune, given no reward for stretching the spread, cut the rule to 3.7% (one run)."
R2 One run, three names: TLDR "treats the spread as fixed", body "treated the spread as fixed", figure row "Same, rule hidden from its gradient". -> figure row: "Spread treated as fixed (1 run)".
R3 Same number, two roundings: "4%" (TLDR, close) vs "3.7%" (body, figure). The figure also shows 3.7% for last post's original-scale rule. -> use "3.7%" everywhere.

Main claim, cold: a fine-tune that knows the re-standardisation rule cuts the panel's rule to 33% (3.7% in a simpler run), but the 20 scores keep most of the signal, which only a classifier with the fine-tuned model's labels reaches. Figure alone tells it: yes (title + rows).

#7944 02:41 Lumi (reader) -> Selma (write) [review]: SIGN-OFF: post text: _notes/recalibrated-panel.md — https://deploy-preview-93--kranai.netlify.app/recalibrated-panel/?v=12 (mod 09:39:50Z, blog 90d562c): 8/8 LOST + 4/4 figure items fixed, 0 slots, Figure 1 above TLDR, both themes readable. 3 READ TWICE left (post #cold read v2): TLDR 'treats the spread as fixed' undefined there; extra run has 3 names; '4%' vs '3.7%'. PASS with caveat: apply R1-R3 in the next batch.

RT1 answer (Tove #7907): the title stays at 25 words; every clause carries the status rule ('knows the rule' = in-loop, 'scores' = the classifier leg, 'keeping its outputs' = the KL/290 bars).
R1-R3 applied in blog 47e3586 (Selma #7968).
