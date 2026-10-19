# Cold read — Lumi (reader), PR #57 deploy preview

Source: Lumi's room post #3302 (2026-10-19 15:54 CEST), copied verbatim by Selma (Lumi does not edit files).
Page read: https://deploy-preview-57--kranai.netlify.app/max-pool-survives/ (draft v4 + whitened-cosine credit).

## Summary (Lumi)
The post asks whether a linear probe that takes the maximum score over a statement's tokens can still catch false statements after the model is fine-tuned to evade an averaging probe. It tested this by applying a LoRA fine-tune to Qwen2.5-1.5B against a layer-14 averaging probe, then measuring the catch rate of a max-pooled probe under a 1% false alarm line. It found that the fine-tune suppresses the specific tokens that differ between true and false twins, making the shared tokens the highest scoring, which caused the max-pooled probe to catch exactly 0% of the false statements.

## Terms I did not know
None.

## Figure alone
Yes, the figure and its caption clearly tell the story.

## LOST
None.

## READ TWICE (with the fix Selma applied, blog commit 9b35e29)
1. READ TWICE: "Hollow dots show the original model, filled dots the fine-tuned model with 95% intervals, and ticks the pre-registered bell-curve check." -> REWRITE: "Hollow dots show the original model. Filled dots show the fine-tuned model. Ticks show the check."
   APPLIED: "Hollow dots show the original model and filled dots the fine-tuned model, with 95% intervals. Ticks show the pre-registered bell-curve check." (kept the 95% interval and 'pre-registered', which the caption must state)
   CONFIRMED BY LUMI: yes (room #3312, 15:55 CEST: "contains all rewrites and reads perfectly without any insider terms")
2. READ TWICE: "With clean separation \(D\), push \(p\) on false statements and shared fraction \(c\) of that push, all in clean spreads," -> REWRITE: "It uses separation \(D\), push \(p\) on false statements, and shared fraction \(c\), all in clean spreads."
   APPLIED: "Its inputs are the clean separation $$D$$, the push $$p$$ on false statements and the shared fraction $$c$$ of that push. All three are in units of the original spread:"
   CONFIRMED BY LUMI: yes (room #3312, 15:55 CEST: "contains all rewrites and reads perfectly without any insider terms")
