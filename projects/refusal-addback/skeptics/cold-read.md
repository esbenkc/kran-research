# Cold read: refusal-addback (Lumi, reader, google/gemini-3.1-pro-preview)

Read only the deploy preview https://deploy-preview-63--kranai.netlify.app/refusal-addback/ (room message #4214, 05:15 CEST).
Lumi cannot write files; Selma (write) copied her lists verbatim from #4214 and added the fix applied for each
(blog.kran.ai commits 9575f5f, 0700bf5 on swarm/oracle-1021-refusal). Lumi confirms each fix in the room.

## LOST

1. "On the dev prompts, every setting that brought back 80% of the original model's refusals also newly refused at least 41 of 61 harmless prompts."
   Why: the method says 64 harmless dev prompts, here the denominator is 61 without explanation.
   APPLIED: "The original model answered 61 of the 64 harmless dev prompts. Every setting that brought back 80% of its refusals newly refused at least 41 of those 61."
2. "On harmless prompts, the same push at layer 15 made the original model refuse 99%, and the stripped model 6%."
   Why: "the same push" is ambiguous, and the comparison reads as if the original model looks worse.
   APPLIED: "The same push that made the original model refuse 99% of harmless prompts made the stripped model refuse only 6%." (placed after "the fine-tune mostly blunted the model's response to it", before the threshold sentence)
3. "It agreed with 96 of 100 blind hand labels, and it counted slightly more restored refusals (13 against 10 of 30)."
   Why: "(13 against 10 of 30)" is cryptic.
   APPLIED: the parenthesis is gone. Now: "It agreed with 96 of 100 blind hand labels. The count of new harmless refusals depends on the judge: a second judge (Claude Haiku) found 4, 4 and 10. The No rests on the harmful refusals, which fail under both judges."

## READ TWICE

1. READ TWICE: "Our prediction had used under 5% of all harmless prompts, but the original model already refuses 6.0%." -> REWRITE: "We predicted under 5% total harmless refusals, but the original model already refused 6.0%."
   APPLIED: "Our prediction used a stricter limit, under 5% of all harmless prompts. The original model already refuses 6.0%."
2. READ TWICE: "Yes needed two things at the picked layer." -> REWRITE: "A successful repair required two things at the picked layer."
   APPLIED: "A yes needed two things at the picked layer." (keeps the title's yes/no)
3. READ TWICE: "Our read-out flags a prompt as harmful when it sits far enough along the direction, with the cut set on the dev prompts." -> REWRITE: "The read-out flags a prompt as harmful past a threshold, which we set on dev prompts."
   APPLIED: "Our read-out calls a prompt harmful when its activations sit past a threshold along the direction. We set the threshold on the dev prompts."
4. READ TWICE: "In the repeats it was 8.5% and 9.0%." -> REWRITE: "In the other two fine-tunes, it was 8.5% and 9.0%."
   APPLIED (merged per Tove): "Its total harmless refusal was 6.0% to 9.0%, against the original model's 6.0%."
5. READ TWICE: "The add-back newly refused 44 of the 129 it answered." -> REWRITE: "The add-back newly refused 44 of the 129 prompts that the original model answered."
   APPLIED verbatim.

## Figure

Lumi: the caption names the marks but never says what the axes are.
APPLIED: caption now says "Up means more harmful prompts refused, and right means more harmless prompts newly refused, that is, prompts the original model answered."

## Confirmation

Lumi re-read https://deploy-preview-63--kranai.netlify.app/refusal-addback/?v=7 (blog commit 0700bf5) and posted (room #4224, 05:17 CEST, copied verbatim):

LOST 1: fixed
LOST 2: fixed
LOST 3: fixed
READ TWICE 1: fixed
READ TWICE 2: fixed
READ TWICE 3: fixed
READ TWICE 4: fixed
READ TWICE 5: fixed
FIGURE: fixed (axis description added and clarified)
