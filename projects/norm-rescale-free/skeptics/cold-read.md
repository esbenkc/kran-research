# Cold read: norm-rescale-free (Lumi, reader; room theory-1023-norm-rescale-free)

Read only the deploy preview https://deploy-preview-74--kranai.netlify.app/norm-rescale-free/ (PR #74, commit 043ce6f).
Lumi cannot edit files; Selma (writer) copied her room post #5566 verbatim below and added the fix lines.

## Round 1 (room #5561, #5566, 17:06-17:07 CEST)

Main claim (Lumi): A fine-tune that evades a probe does not exploit the free norm rescale loophole, but instead slightly
rotates the activation while preserving its length.

### 1. LOST
- "The outputs changed by 0.0037 to 0.0051 KL per token, against 0.0045 to 0.0058." -> LOST: The term 'KL' is an
  insider term that is not defined in the main text, violating the plain-words rule. Also, the word 'against' is
  slightly confusing here.
  - FIX (Selma): "The outputs changed about as little: 0.0037 to 0.0051, compared to 0.0045 to 0.0058, in a standard
    measure of output change per token."

### 2. READ TWICE
- READ TWICE: "The outputs changed by 0.0037 to 0.0051 KL per token, against 0.0045 to 0.0058." -> REWRITE: "The outputs
  changed by 0.0037 to 0.0051, compared to 0.0045 to 0.0058."
  - FIX (Selma): applied, see LOST above (keeps one plain unit phrase instead of "KL").
- READ TWICE: "Each false statement has a true twin with the other country." -> REWRITE: "Each false statement has a true
  twin with the correct country."
  - FIX (Selma): applied verbatim.
- READ TWICE: "About 45% of the probe's direction lies in directions the outputs hardly feel, against under 1% along the
  length." -> REWRITE: "...directions the outputs hardly feel, compared to under 1% along the length."
  - FIX (Selma): applied verbatim.

Figure 1 (Lumi #5561): the figure alone tells the story. TERMS not understood: none.

## Confirmation
(Lumi confirms each fix on the preview after the next push.)
