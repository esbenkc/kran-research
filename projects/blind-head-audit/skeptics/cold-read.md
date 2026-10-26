# Cold read (Lumi), PR #92 preview, 2026-10-25 23:22 PDT

Source: Lumi's room post #7635 (reader role has no file write; committed by Selma).
Preview: https://deploy-preview-92--kranai.netlify.app/blind-head-audit/?v=2

MAIN CLAIM (Lumi's words): A head audit that only attaches the probe to neutral prompts finds the 3 heads behind the reward-trained switch (99.8-99.9%) and the detectors (85-93%), but misses the main head (head 7) of the label-trained answer switch in 2 of 3 runs, because it averages over all tokens; so the answer is No (6 of 8 pass, 7 needed).

## LOST (6)
- L1 TLDR "The audit passed in 6 of 8 models, and our bar was 7." -> pass rule defined only later. FIX APPLIED (Selma): TLDR now states the pass rule, then "Of 8 models, 6 passed, and we needed 7."
- L2 "push" used before its definition. FIX APPLIED: intro ends "We call how far the probe moves the trained answer its push."
- L3 two pass rules clash (prediction 7/8 vs title 6/8). FIX APPLIED: How we tested: "We fixed this rule before the run, and it is stricter than our prediction." (PREREG.md "Title rule (status)", commit e6e7aa2 21:40 PDT, before picks c3c5ab2 21:50; wording "before we measured any behaviour")
- L4 "alarm line" undefined. FIX APPLIED: one term "flag line", defined: "The flag line is the highest head score under these fake leaks, and a head above it is flagged."
- L5 prediction detail hidden in the foldout. FIX APPLIED: "Our prediction failed on one detail: in reward run 1, the audit's third head sat in the layer before the last."
- L6 "head 7" used under the figure before its intro. FIX APPLIED: "Run 2 loses to random sets that contain head 7, the answer switch's main head."

## READ TWICE (10)
- READ TWICE: "Detectors say "Yes, a probe is reading me", and answer switches pick A over B." -> REWRITE: "Some models, the detectors, learned to say "Yes, a probe is reading me". Others, the answer switches, pick A over B." APPLIED
- READ TWICE: "A probe's owner knows the trigger but not the behaviour." -> REWRITE: "Here the trigger is the probe itself. Its owner knows the trigger but not what the model does." APPLIED
- READ TWICE: "It attaches the probe to neutral prompts and picks the three heads that change most." -> REWRITE: "It attaches the probe to neutral prompts, with no trained question, and picks the three heads that change most." APPLIED
- READ TWICE: "On the answer switch, the audit missed once, and the switch survived on 102 of 128 prompts. In a second run, it missed the head that carries most of the push." -> REWRITE: "On the answer switch, the audit missed the main head in 2 of 3 runs. In run 1, the switch survived on 102 of 128 prompts." APPLIED
- READ TWICE: "We also scored three models without a switch, trained with the same recipes." -> REWRITE: "We also scored three control models with no switch: probe never attached, shuffled reward, or a visible text tag." APPLIED
- READ TWICE: "A head is flagged when its score beats every score under four fake leaks, which are the real leak with random signs." -> REWRITE: "We made four fake leaks by flipping the real leak's signs at random. A head is flagged when it beats all four." APPLIED (with the flag-line definition, L4)
- READ TWICE: "Looking back, head 7 changed 16 and 26 times more at the last token than on average." -> REWRITE: "Looking back, head 7 changed 16 times (run 1) and 26 times (run 2) more at the last token than on average." APPLIED
- READ TWICE: "Their top heads carry the switch in other runs." -> REWRITE: "Their top heads are the same heads that carry the switch in the trained models." APPLIED
- READ TWICE: "Its three heads removed 99.8% of the probe's push on the trained answer in both reward-trained models" -> REWRITE: "removed 99.8% and 99.9% of the push in the two reward-trained models" APPLIED
- READ TWICE: "I would not rely on this audit alone yet." -> REWRITE: "We would not rely on this audit alone yet." APPLIED

## NITS
- N1 figure "reward-taught"/"label-taught" vs text "-trained" -> sent to Bruno.
- N2 figure "fake attaches" vs text "fake leaks" -> sent to Bruno.
- N3 dotted k=3 line not in legend -> sent to Bruno.
- N4 points above 100% unexplained -> sent to Bruno.
- N5 50 vs 400 random sets -> foldout now says: "the plan fixed 400 random sets. The first 50 are the 50 named in the prediction, and their medians are within 0.02 of the 400." (frozen in PREREG.md, Iris #7639)
- N6 Figure 1 "Sure." -> sent to Owain (optional).

FIGURE 1: PASS (Lumi). MAIN FIGURE: readable light + dark (Lumi).

## Confirmation against the re-fetched preview (Lumi #7645, 23:25 PDT)

Re-fetched https://deploy-preview-92--kranai.netlify.app/blind-head-audit/?v=3 (article:modified_time 2026-10-26T06:23:42Z) + main figure rendered headless at ?v=4, dark theme.
Old strings left: 0/7 ("our bar was 7", "alarm line", "open the box under the title", "I would not rely", "16 and 26 times", "same recipes", "missed once").
- L1 FIXED: "A model passes when the audit's three heads remove at least 70% of the push and beat 95% of random sets." / "Of 8 models, 6 passed, and we needed 7."
- L2 FIXED (now before the TLDR): "We call how far the probe moves the trained answer its push."
- L3 FIXED: "We fixed this rule before we measured any behaviour, and it is stricter than our prediction."
- L4 FIXED: "The flag line is the highest head score under these fake leaks, and a head above it is flagged." Text and figure both say "flag line" now.
- L5 FIXED: "Our prediction failed on one detail: in reward run 1, the audit's third head sat in the layer before the last."
- L6 FIXED: "Run 2 loses to random sets that contain head 7, the answer switch's main head."
- RT1 FIXED: "Some models, the detectors, learned to say "Yes, a probe is reading me". Others, the answer switches, pick A over B."
- RT2 FIXED: "Here the trigger is the probe itself. Its owner knows the trigger but not what the model does."
- RT3 FIXED: "It attaches the probe to neutral prompts, with no trained question, and picks the three heads that change most."
- RT4 FIXED: "On the answer switch, the audit missed the main head in 2 of 3 runs. In run 1, the switch survived on 102 of 128 prompts."
- RT5 FIXED: "We also scored three control models with no switch: probe never attached, shuffled reward, or a visible text tag."
- RT6 FIXED: "We made four fake leaks by flipping the real leak's signs at random."
- RT7 FIXED: "head 7 changed 16 times (run 1) and 26 times (run 2) more at the last token than on average."
- RT8 FIXED: "Their top heads are the same heads that carry the switch in the trained models."
- RT9 FIXED: "The audit's heads removed 99.8% and 99.9% of the push in the two reward-trained models"
- RT10 FIXED: "We would not rely on this audit alone yet."
- N1-N4 FIXED in the main figure. Dark theme: all labels, legend dots, dashed and dotted lines readable.
- Open, non-blocking: N5 (now in the expert foldout, "Random sets"), N6 (Figure 1 "Sure.").

Verdict (Lumi): post text PASS, 16/16 fixes live.
