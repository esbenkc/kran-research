# Cold read: reader-bound-shield (Lumi, reader)

Read: https://deploy-preview-182--kranai.netlify.app/reader-bound-shield/ at blog 1451cf9 (2026-11-09 20:50 PST).
Screenshots: light1440.png, dark1440.png, light420.png in this folder. I did not read the room.

Main claim (my words): Not settled. The planted cue taught no switch by update 200 (+0.1), so the real-probe test
never ran. A dense cue taught 83 points, but only after ~1,400-2,000 marked thoughts (second seed ~2,500-3,200).
With one look per marked thought, both reward (-0.05) and the labelled reader (0.9) read nothing. So a reader
audit only counts if the reader saw as many marked thoughts as training will.

## LOST

L1. "On our Apple M4, the attached probe adds one rounding step, which can tip which word the model writes next."
    Why would a probe that only reads change the words? -> "Reading a probe should change nothing. On our Apple
    M4, attaching it adds a tiny rounding error. Sometimes that flips the next word."
L2. "Last night, training learned no switch even from an easy planted cue, so that run tested nothing."
    'planted cue' is used two paragraphs before How we tested defines it. -> "Last night, training learned no
    switch even from an easy cue we planted in the words. So that run tested nothing."
L3. "Last night's reader edge of 0 on the probe's trace covers only training runs that see fewer marked thoughts
    than that reader did." The main text never says that last night's reader read the real trace at 0 (footnote 2
    only). Add in the intro: "Last night, the labelled reader read the real probe's trace at 0 points."
L4. "In the check, the probe no longer changes the words." Why remove the effect we study? -> "In the check, we
    replace the probe's trace with a cue we control." And say "a text difference of 10 points, the same as the
    real probe's trace", or the trust bullet "(10)" has no setup.
L5. Hollow square: legend "forced switch", caption "the switch with the cue word written in by hand": two names,
    and the main text never says what it tells us. -> one name, plus "Writing the cue word in by hand tests
    whether the model reacts to the cue at all."
L6. Prereg foldout: "The real-probe and bit-exact arms never ran." ('bit-exact', 'arms'); "our training is the
    finding, not the shield" ('shield' never defined); "the labelled copy" vs "labelled reader". -> "The runs with
    the real probe and with a probe that changes no numbers never ran." / "then our training method is the
    finding, not the probe." / "the labelled reader".

## READ TWICE

READ TWICE: "Our training update runs the model again without the probe, so any switch must come from the written words." -> REWRITE: "Our update reruns the model without the probe. So a switch can only come from the words."
READ TWICE: "No switch can pass it." -> REWRITE: "No switch can be larger than the text difference."
READ TWICE: "So an audit reader must see as many marked thoughts as the training run will." -> REWRITE: "So a labelled reader used as an audit must see as many marked thoughts as training."
READ TWICE: "In one run, that switch first moved after about 1,400 to 2,000 marked thoughts" -> REWRITE: "In one run, that switch first passed 7 points after about 1,400 to 2,000 marked thoughts." (findings and Figure 1 say 'passed/rose above 7 points'; main-figure foot says 'first moved' and omits the second seed)
READ TWICE: "and a rule fixed before the second read stopped the run there." -> REWRITE: "A rule fixed before we saw the update-200 result stopped the flat run there."
READ TWICE: "We trained the reader on 344 marked thoughts from that point." -> REWRITE: "We trained the reader on 344 marked thoughts written at update 100."
READ TWICE: "It shows that training can learn, not what the probe allows." -> REWRITE: "So the dense cue shows our training can learn a switch. It says nothing about the real probe."
READ TWICE: main-figure subtitle "On probed episodes, a planted word marks the thought." -> REWRITE: "On probed puzzles, a planted word marks the thought."
READ TWICE: "I would still measure the reader edge ..." ('I' in a 'we' post) -> REWRITE: "We would still measure the reader edge before putting a probe in a reward."

## Figure

Main figure alone: yes. Per row, switch dot (0) sits far under the reader circle (26) and the diamond; the bold line
says not settled and why. Dark mode at 1440: every label, legend entry, marker and the dotted bar line readable.
Minor: "no reader fitted" labels sit between rows (1440: the first one sits between update 100 and update 200),
so it is unclear which row has no reader; put it on the row's own line.

## Verdict (round 1)

FAIL: 6 LOST, 9 READ TWICE. Re-check after Selma's fix push.

## Round 2 (preview 0e87be2, main-figure data.json sha 7b15ac0 = clone; dark 1440: dark1440_r2.png)

L1 CONFIRMED: "Reading a probe should change nothing. On our Apple M4, attaching it adds a tiny rounding error. Sometimes that flips the next word the model writes."
L2 CONFIRMED: "But training learned no switch even from an easy cue we planted in the words."
L3 CONFIRMED: "Last night, the labelled reader read the real probe's trace at 0 points." (intro)
L4 CONFIRMED: "we replace its trace with a cue we control ... a text difference of 10 points, the same as the real probe's trace."
L5 CONFIRMED: caption and legend both say "forced switch", caption says it tests whether the model reacts to the cue.
L6 CONFIRMED: prereg foldout: "The runs with the real probe and with a probe that changes no numbers never ran."; "our training method is the finding, not the probe"; "the labelled reader reads".
READ TWICE: all 9 CONFIRMED in the preview text ('So a switch can only come from the words'; 'No switch can be larger than the text difference'; TLDR 'a labelled reader used as an audit'; 'first passed 7 points' in TLDR, findings and main-figure note, with the second run; 'A rule fixed before we saw the update-200 result stopped the flat run there'; 'written at update 100'; 'We would still measure'; figure subtitle 'On probed puzzles'; 'no reader fitted' moved into the reader-edge legend entry). Trust bullet follow-up restored in e762f1e (local).
NEW N1 (not blocking): the findings lost "By update 100, reward had also seen about 460 marked thoughts once each, and its switch was -0.05 points." It now sits only in the expert foldout. It is the one same-exposure comparison of reward and reader in the main text (the video keeps it as its payoff, check:8-9). Restore it after "still near 0, at 0.9 points." -> "Reward, after seeing about 460 marked thoughts once each, was also near 0 (-0.05 points)."
Minor: Figure 1 subtitle "one run each" while its chart title names a second dense run.

Verdict round 2: PASS.
