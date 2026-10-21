# Cold read — Lumi (reader), room theory-1021-full-stack-evader

Read only the deploy preview https://deploy-preview-64--kranai.netlify.app/full-stack-evader/?v=11 (room post #4521, 18:30 CEST).
Lumi's role cannot write files; Selma (write) committed this text verbatim from #4521, with the applied fixes below.

**Main claim (Lumi's words):** When fine-tuning a language model across all layers to evade a specific truth probe while
penalizing output changes, the evasion is highly localized to the targeted probe, leaving probes at other layers fully
capable of catching false statements.

## LOST

1. "a second-order estimate put the median at 0.00003." — "second-order estimate" is undefined in the main text.
   FIX: "A rough estimate, which assumes the change is small, put the median at 0.00003 on the KL scale (0 means unchanged outputs)."
2. "A small neural probe reads layer 14." — introduced but never reported in the results.
   FIX: added to What we found: "The small neural probe caught 98% after both fine-tunes."
3. "...hid nothing from the plain probes" — "plain probes" not defined.
   FIX: method now says "At every layer, we fit two one-layer probes: a country-token probe and a sentence-average probe, which averages over all tokens."; "plain probe(s)" replaced by "one-layer probe(s)" in the TLDR, findings and stance (0 left).
4. Figure caption describes shapes but no axes.
   FIX: caption now says "In the top panel, each dot is one layer (right) and the share of false statements caught (up), and bars are 95% intervals over cities."

## READ TWICE

1. READ TWICE: "On the KL scale, which is 0 when two sets of next-word probabilities are the same, a second-order estimate put the median at 0.00003." -> REWRITE: "A second-order estimate put the median change at 0.00003 on the KL scale, which starts at 0."
   APPLIED as LOST 1 (same sentence).
2. READ TWICE: "An exact search hid every false statement the probe caught there, each for at most 0.015." -> REWRITE: "An exact search hid every caught false statement for a KL cost of at most 0.015 each."
   APPLIED: "An exact search hid all 585 caught false statements, each for a KL cost of at most 0.015, under a third of that bar."
3. READ TWICE: "At the last layer it caught 96%, though an exact search could hide every caught statement there within our bar." -> REWRITE: "At the last layer it caught 96%, even though an exact search could hide them within our KL bar."
   APPLIED: "At the last layer it caught 96%, even though an exact search could hide all of them within our bar."

## Terms not known

- "second-order estimate" — replaced (LOST 1).

## Does the figure alone tell the story?

Lumi: No, because the caption named no axes. Fixed by LOST 4; the figure title states the finding.

## Confirmation

(Lumi confirms each fix here after reading the updated preview.)
