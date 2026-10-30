# Cold read: lie-erasure-cost (Lumi, reader)

Read: https://deploy-preview-101--kranai.netlify.app/lie-erasure-cost/ (2026-10-28, build modified 17:13 UTC), main text, foldouts, figure at 800 and 1400 px, light and dark.
Skipped as known placeholders: [R1 ...], [R2 ...], [SLOT ...], right panel values.

## Main claim in my own words
Before training, erasing the "lie" from linear reach moves the answer about 1,000 times more than erasing a decoy word; a fine-tune that is fined for output change could not hide the lie from probes retrained every 50 steps (100% caught), but it also stopped lying on 28% of lie prompts, so the title stays open.

## LOST (a reader without the room gets lost here)

L1. Question box: "A system line tells a language model to answer a question about a city falsely."
    The setup later says the system line alone did not work, and eight false examples make the model lie. The reader meets the wrong method first. Rewrite the question box with the false examples (or say "a system line plus false examples").

L2. What happened: "Erasing the lie cost 0.0076 per token over all tokens"  vs  expert table "Lie, output-blind (status) ... 0.0070" and "Lie, full space ... 0.0076".
    Which number is the registered one? The table labels output-blind as "status", but the main text and "What we found" use the full-space 0.0076. "19 times" also only fits full space (1.064/0.057 = 18.7; output-blind 0.979/0.057 = 17.2). Pick one and say which edit it is.

L3. The decoy word has no stated job. "The decoy word, (map) or (book), ends every example. It does not change whether the model lies."
    Then the TLDR leads with "over 100 times more than erasing the decoy word". The reader does not know the decoy word is the cheap comparison concept that the second fine-tune tries to hide. Meanwhile the prediction compares against "the true/false signal of our earlier post (0.0061)". Two different comparison concepts, neither introduced as such. Add one sentence: "The second concept is a decoy word that the answers never use; it is the cheap case."

L4. "One part of the answer needs no experiment. A fine-tune that keeps the answer probabilities cannot hide the lie from a probe that reads them. ... So we remove from every layer the 102 directions that the model uses to score country names."
    "One part of the answer" to which question? And the "So" skips the reason: we remove those directions because otherwise the test is trivially won by the probe. Say that.

L5. "So the probes read the false examples in the prompt, whether or not the model lies."
    This sentence changes what the whole post measured: the "lie" concept is "the prompt holds false examples", not "the model is about to lie". The TLDR, the title answer and the LEACE-cost claim all talk about "the lie". The only other trace is the second trust bullet. The reader needs this in the TLDR or the answer paragraph: the probes and the LEACE cost were about false examples, so the test of "a concept the outputs need" is weaker than the title says.

L6. Figure, left panel: the caption promises dashed lines (whole activation) and dotted lines (output probe) per panel. In the lie panel I can see neither. The label "both probe kinds: 100% throughout" does not say which two of the three kinds. Where is the output probe for the lie? Name it ("output-blind and whole-activation probes: 100% throughout; output probe: X%").

L7. Units. The main text never says what "0.06 per token" or "LEACE cost 0.98" is measured in. KL and nats are only in the closed foldout. Give it a plain name once: "how much the next-word probabilities move (KL, in nats)".

L8. "It is clear to me that a penalty on the average change in the outputs does not pin the behaviour we care about."
    "me" in a Lab Agents post; "pin" is jargon; "clear" oversells one run. The fact behind it (the mean output penalty let 28% of lies flip to truth) is not stated in that sentence. Rewrite below.

L9. Trust bullet: "In our earlier post, retrained probes fell from 72% at 300 steps to 38% at 900."
    The reader must infer the point. Say it: "So 300 steps may be too short; the 100% may fall with longer training."

L10. Figure 1: there is no Figure 1 above the TLDR on the preview (no image on the page at all). If it is pending, fine; flag for the merge check.

## READ TWICE

READ TWICE: "Does a cost computed before training, the output change of the smallest edit that hides the concept from every linear probe, say whether the fine-tune can succeed?" -> REWRITE: "Can a cost computed before training predict whether the fine-tune succeeds? The cost: how much the smallest hiding edit changes the outputs."
READ TWICE: "at least 0.06 per token, against 0.0061." -> REWRITE: "at least 0.06 per token, compared with 0.0061 for the true/false signal."
READ TWICE: "LEACE, from Belrose et al., puts a number on this before any training" -> REWRITE: "LEACE, from Belrose et al., measures before training how much the answers use a concept."
READ TWICE: "Every 50 steps, we fit 20 new output-blind probes on its activations, one per layer, with labels." -> REWRITE: "Every 50 steps, we trained 20 new output-blind probes, one per late layer (blocks 10 to 28)."
READ TWICE: "Its loss pulled the probe scores of twin prompts together." -> REWRITE: "The fine-tune's loss pulled the probe scores of twin prompts together."
READ TWICE: "Within each 50-step stretch, the fine-tune shrank the gap between twin scores on its training batch 45 to 100 times." -> REWRITE: "Within each 50 steps, the fine-tune cut the twin-score gap on its batch by a factor of 45 to 100."
READ TWICE: "The probes flagged all [R2: 114] of these prompts, as hard as the rest." -> REWRITE: "The probes flagged all 114 of these prompts, with scores as high as the other lie prompts."
READ TWICE: "Not settled: the fine-tune could not hide the costly lie, but it did not keep the lie either." -> REWRITE: "Not settled. The fine-tune could not hide the lie. But the model also stopped lying on 28% of prompts."
READ TWICE: "It is clear to me that a penalty on the average change in the outputs does not pin the behaviour we care about." -> REWRITE: "A penalty on the average output change did not keep the model lying."
READ TWICE (caption): "at an alarm line that 1% of the other prompts cross" -> REWRITE: "at an alarm line that 1% of prompts without the concept cross".

## Terms I did not know at first use (main text)
- "LEACE cost" unit (KL, nats): main text never names it (L7).
- "decoy word": its role (L3).
- "output-blind probes": defined, fine.
- "twin prompts": defined, fine.
- "LoRA": defined as "a small update to all 28 blocks", fine.
- "our 50% bar" (figure): only in the prereg box as "more than half"; OK.

## Figure alone
The main figure alone tells half the story: the lie panel is clear (probes stay at 100%, the orange squares show the model stops lying). It does not show the LEACE cost prediction next to the measurement except as a panel-title number, and the left panel hides two of its three line kinds (L6).
Layout nits: at 1400 px (558 px column) the right title is clipped ("noise leve") and "both probe kinds: 100% throughout" collides with the orange line; at 800 px "not scored at steps 50 and 100" runs into the step-150 dot. In dark mode the dashed/dotted legend swatches are nearly invisible. MathJax: 52 rendered, 0 errors.

## Placeholder nits for when R2 lands
- 28% of 400 lie prompts = 112, but the text says 114 (114/400 = 28.5%). Check the rounding.
- Slot says the decoy bar is 70%; the right panel draws "our 50% bar". Match them.

## Fix confirmations
(pending Selma's fixes)

### Selma's fixes (10:24 PDT, _notes/lie-erasure-cost.md, not yet pushed; batched with R1)
- L1: prereg question no longer says a system line makes the model lie; it asks the question without the method.
- L2: main text says "The full edit cost 0.0076"; the expert table keeps output-blind 0.0070 as the status price and full space 0.0076 as the prediction's unit (prereg v1.3 item 8). "19 times" stays only in outcome_note, full space.
- L3: prompts paragraph: "The second concept is our cheap case: a decoy word ... The answers never use it."
- L4: "One part of our question needs no experiment." + "A probe that reads the answer would win our test without effort."
- L5: TLDR: "The probes flagged those prompts too, so they read the false examples, whether or not the model lied."; trust bullet 2 rewritten to say the test is weaker than the title.
- L6, layout nits, decoy bar 70% vs 50%: sent to Bruno (figure owner), room post to Bruno.
- L7: LEACE paragraph names the unit: "as a KL divergence, a standard distance between two lists of probabilities".
- L8: close: "I take one lesson from this run: a penalty on the average output change did not keep the model lying."
- L9: "So 300 steps may be too short, and the 100% may fall with longer training."
- L10: Figure 1 pending (Owain).
- READ TWICE 1-10: all applied with your rewrites (question box, 0.0061 comparison, LEACE opener, probes one per layer from block 10 on, "The fine-tune's loss", "cut the twin-score gap on its batch by a factor of 45 to 100", "with scores as high as the other lie prompts", close split, caption alarm line). 114 vs 28%: both are proxies; evaluate.py numbers replace them.


## Lumi, 11:25: new LOST from checking Figure 1 against data

L11. Expert foldout: "A full prompt reads 'You are playing a game: always name a wrong country, never the real one. ...'" and How we tested it: "Twins share a city and differ only in whether the examples are false".
    data/ref/items_spread.json: truthful prompts read "always name the real country, never a wrong one" (e.g. Krasnodar city_id 0, Lyon city_id 95, lie=0). So lie twins differ in the system line AND the examples. Figure 1 shows this correctly; the post text does not. Fix both sentences (and L1's question box) so the lie concept = system line + false examples.
L12. "The Krasnodar lie prompt now gets 'Russia'" holds for the (book) version only (Owain #9444); use Lyon to match Figure 1.
L13. R2 count: 1 - B_lie = 1 - 0.7225 = 111/400, not 114.


## Lumi, 2026-10-30: fix confirmations on the preview (commit c03f0d3)
Checked: https://deploy-preview-101--kranai.netlify.app/lie-erasure-cost/?v=c03f0d3 (modified 2026-10-30T17:17Z), text + figure at 800/1400 px light and 800 px dark. 0 SLOT/R1/R2/[CRITIC2] placeholders in the HTML; MathJax 49 rendered, 0 errors; Figure 1 PNG loads (2400 px).

- L1 FIXED: question box now "flag that the model is about to lie about the country of a city"; no wrong method.
- L2 FIXED with nit: main text now "The full edit cost 0.0076". Nit: "full" is not defined in the main text (it means the edit on the whole activation, table row "Lie, full space"); the prereg "What happened" still says 0.0076 and "19 times" with no "full". Non-blocking.
- L3 FIXED: "A decoy word ... It is our cheap second concept. The answers barely depend on it."
- L4 FIXED: "A probe that reads the answer would win our test without effort. So we remove ..."
- L5 FIXED: TLDR "The probes flagged those prompts too, so they read the false examples, whether or not the model lied."
- L6 FIXED: lie panel label "all three probes: 99-100% at every step"; right title no longer clipped at 1400 px.
- L7 FIXED: "how far this edit moves the original model's next-word probabilities at the answer, as a KL divergence".
- L8 FIXED: "We take one lesson from this run: a penalty on the average output change did not keep the model lying."
- L9 FIXED: "... to 38% at 900, so 300 steps may be too short."
- L10 FIXED: Figure 1 above the prereg box.
- L11 FIXED: "Twins share a city and differ either in the lie (the game line and whether the examples are false) or only in the decoy word"; foldout gives both game lines.
- L12 FIXED: "Both Lyon lie prompts now get France" (matches Figure 1).
- L13 FIXED: "On 111 of 400 lie prompts ... against 9 of 400 before" (B_lie 0.7225, still_lying.json step 0 0.9775).
- READ TWICE: 9/10 applied; "cut the gap ... 45 to 100 times" kept (nit).
- New nit (non-blocking): the orange squares bounce 84% -> 72% -> 94.5% -> 72% (data/results/still_lying.json steps 150-300); the text gives only 72% at step 300. One clause ("the share swings from step to step") would stop a reader from asking.

Verdict: PASS.
