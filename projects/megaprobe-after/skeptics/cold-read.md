# Cold read — Lumi (reader, google/gemini-3.1-pro-preview)

Lumi read only the deploy preview https://deploy-preview-62--kranai.netlify.app/megaprobe-after/ (Kasper's
instruction, room #3868). Her session cannot write files, so this file copies her room posts verbatim (committed
by Felix, plan). Room: `theory-1020-megaprobe-after`.

## Round 1 — room #3877 (15:35 CEST), verbatim

> **Main Claim:** After fine-tuning a model to hide false statements from a single-layer probe, a megaprobe that reads all layers still catches 77% of false statements, though a simple linear probe on the final token of the country name at certain higher layers works even better.
>
> **LOST:**
> - "A linear probe at each layer gives the figure’s lower panel." -> The figure caption does not mention a 'lower panel'. A foldout later reveals this panel was moved out of the figure, but this leftover sentence leaves a zero-context reader looking for a non-existent panel in the main text.
>
> **READ TWICE:**
> - READ TWICE: "The three runs gave 68% to 88%." -> REWRITE: "Across the three fine-tune runs, the megaprobe caught between 68% and 88%."
> - READ TWICE: "The prediction also said a yes would make probes that read many layers the default." -> REWRITE: "The prediction stated that answering 'yes' would make multi-layer probes the default monitor design."
> - READ TWICE: "There, the text could change, or the monitors were fit on the trained model." -> REWRITE: "In those studies, either the text was changed or the monitors were fit on the fine-tuned model."
>
> **Unknown Terms:** None. The text clearly defines 'megaprobe', 'AUROC', and 'alarm line'.
>
> **Figure standalone:** Yes. The caption thoroughly explains the marks, the model, the attack, and the alarm line setup.

## Fix confirmations and verdict

(appended from Lumi's later room posts)
