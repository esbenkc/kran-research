# Cold read: tell-chain-8b (Lumi, reader)

Read: https://deploy-preview-25--athanorscience.netlify.app/tell-chain-8b/ (served HTML 54,636 B, 2026-11-27 07:46 PST).
Main text (TLDR to "independent costs", incl. main-figure caption) = ~890 words; target 400-500.

Main claim in my words: after a Qwen3-8B tail (last 2 layers) has learned watched-only switches for three cues,
it learns a fourth in about a fifth of the RL updates of matched controls (0.19, seed-level 0.17-0.23),
below both pre-run models (multiplying 0.41, flat 0.74), so the 0.55 test passes.

Rebuilt from the expert table: geomean main/controls cue C 0.322, cue D 0.195; main D/C 0.673. All match the post.

## LOST
L1. Same quantity, two numbers. Prose: "Our last run measured one step: with one switch learned, the next cost 0.74 of the training."
    Expert foldout: "Cue B (last run) vs coin-flip controls 0.768 (0.690 to 0.855), vs fresh 0.731".
    Main figure: hollow pink mean sits near 0.77, but both model lines start at 0.74.
    A reader cannot tell which number the lines are built from, or why the controls' value is not 0.74.
    Fix: one sentence that says 0.74 is the last post's number (and against which reference), and that the lines start there.
L2. Three references for the 0.55 test.
    Question line: "...the fourth tell costs at most 0.55 of a fresh tail's updates?"
    Prereg 'What we expected': "at most 0.55 of the training of the same-training controls".
    Body: "both intervals lie wholly below 0.55 ... Our rule ... calls that a yes" (both references).
    Intro: "does the fourth cost at most 0.55 of the training?" (no reference).
    Fix: the body and prereg foldout say "against both the controls and fresh tails" each time. (The question: line is verbatim; leave it, but the intro question should name both.)
L3. "Prediction held" vs "both lines missed". TLDR: "Both lines we drew before the run missed, because the cost fell faster than either."
    Prereg: "It held." A reader reads that as a contradiction. The 'lines' are never called models in the TLDR.
    Fix: say the prediction was a threshold and the lines were two models: "Our test (below 0.55) passed. Both models we drew still missed: the cost fell faster than either."
L4. One-switch group logic. "The tails that had learned only cue B needed 0.35 times the controls' updates (0.27 to 0.45). So one earlier switch on a cue of the same size gave most of the drop. In our last run, the switch on the larger cue A gave only 0.74."
    The reader must compare 0.35 with 0.32 from two sentences back, and must guess that 'the drop' means 0.32.
    This paragraph is also the evidence that cuts against 'multiply': cue A added almost nothing at cue C (secondary read 0.91, not settled). Say that plainly next to the Yes.

## READ TWICE
READ TWICE: "Both lines we drew before the run missed, because the cost fell faster than either." -> REWRITE: "Our 0.55 test passed. The cost fell faster than both models we wrote down."
READ TWICE: "At the third tell, one earlier switch on a cue of the same size already gave most of that drop." -> REWRITE: "Tails that learned only cue B were almost as fast at cue C: 0.35 against 0.32."
READ TWICE: "The controls get the same number of updates with no switch to learn: their praise penalty follows a coin flip, and they see no cue." -> REWRITE: "The controls get as many updates, but no cue. A coin flip decides when praise is penalised."
READ TWICE: "We count the updates until praise openers are 10 percentage points rarer with the cue on, at two checks in a row." -> REWRITE: "We count updates until praise openers are 10 points rarer with the cue on than off, twice in a row."
READ TWICE: "Our read: in this tail, the first switch on a small cue does most of the work, and the next switch still cut the cost by a third." -> REWRITE: "Our read: the switch on cue B, the first small cue, does most of the work. Cue C's switch cut a further third."
READ TWICE: "Yes: the fourth cost about a fifth, though not by a fixed factor per tell." -> REWRITE: "Yes: the fourth cost about a fifth. The saving did not shrink by one fixed factor per tell."
READ TWICE: main-figure caption "divided by those of coin-flip controls (left) or fresh tails (right)" (reads as two panels) -> REWRITE: "left dot of each pair: vs coin-flip controls; right dot: vs fresh tails"
READ TWICE: "A prompt-level interval instead resamples the 64 check questions. It gives 0.16 to 0.21 against the controls." -> REWRITE: "Resampling the 64 check prompts instead of seeds gives 0.16 to 0.21 against the controls."

## Terms
Four words for one idea: tell, cue, (activation) trigger, switch. Title says "Triggers", question "tells", body "cue"/"switch".
Body mixes "fourth tell" with "cue D". Suggest: cue = the planted vector, switch = the learned behaviour, and say once "the 4th tell = cue D".
Not defined in main text: LoRA (only a footnote; one plain clause would do), "orthogonal" (question line; body says "at right angles", fine).
Undefined but guessable: "fresh tails", "coin-flip controls", "seed-level interval".

## Figures
Figure 1 alone tells the story: yes. Head (64 vs 320 = 0.20) and bars match the expert table for seed 8 (C 80/320, D 64/320); ticks 176/237 and 131/237 match 0.55, 0.74, 0.41 x 320.
Minor: legend "control with no earlier cue" reads like an untrained model; suggest "control: same updates, no earlier cue".
Main figure (thumb.png checked, current, 5 dots per arm): readable; see caption and L1.

## Fix check (Selma 6a0e8ae)
Netlify preview had not rebuilt at check time (served 54,836 B, old strings 6/6 still present; only figure-lint listed in gh pr checks).
Checked on git show origin/swarm/theory-1127-tell-chain:_notes/tell-chain-8b.md (diff 2ccd29f..6a0e8ae) and the committed thumb.png.
- L1 FIXED: "about three quarters of the training (0.73 to 0.77, depending on what we compared with)" + "we built two outcomes from its headline value, 0.74".
- L2 FIXED in body/intro/caption: "at most 0.55 of a comparison tail's training"; caption "0.55, which the 4th tell had to beat against both". Left as is: prereg 'expected' names only the controls (prereg text, non-blocking).
- L3 FIXED: TLDR "The cost fell faster than both outcomes we wrote down."
- L4 FIXED: "almost as fast: 0.35 ... against 0.32 for the main tails. So the extra earlier switch, on cue A, added little here." TLDR says the number of earlier switches mattered little at the 3rd tell.
- READ TWICE 1-8: all applied (coin flip split, "than off", stance names cue B, loop-closer split, caption left/right dot, prompt-level rewrite).
- Terms: tell defined ("a hint that a grader is looking"), "each tell is one cue: the 2nd tell is cue B ...", LoRA "a low-rank add-on to its weights".
- Main figure thumb: new title "below both pre-run predictions", legend bar says "against both comparison tails". Readable.
Open, non-blocking: main text still ~880 words (over 500 needs a PR line).
