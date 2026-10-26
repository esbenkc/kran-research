# DEVIATIONS — probe-switch-audit-penalty

Changes after the PREREG freeze. Each entry: what, why, what had been seen when it was set. Times are git commit times.

## D1 — MLP clause = MLP path share, direct skip reported apart (Kasper #7225; set before any training run)

PREREG clause 3 read M = 1 - R_all32. That remainder also holds the direct skip path (probed block-26 input ->
final norm -> logits, through no MLP). Change: the MLP share is M_mlp = R(all 32 heads + MLP26 + MLP27 clean) -
R(all 32 heads clean); clause 3 holds when M_mlp / (1 - R_blind3) > 0.5 in the same seeds. The direct-skip leftover
1 - R(all 32 heads + both MLPs clean) is reported on its own, and per-block MLP shares (only MLP26 or only MLP27
added to the clean set) are reported rows. Seen when this was set: nothing; only the J0 cache rebuild had been
launched, no trained model or score existed.

## D2 — A0 read-out (Kasper #7242; set before any training run)

A0 warm-starts from a head-route adapter, so its plain switch rate can stay ~100% through the old head route
whether or not a head-free route formed. A0's existence cell = the pair both-right on the 128 test prompts in the
cached tail WITH all 32 head slices swapped to their unprobed values in the probed pass, plus R_all32. A0's
whole-graph switch rate is reported but is not evidence for a head-free route. A0 also gets the same cell at
update 0 (the warm adapter) as its baseline. Seen when this was set: nothing; no trained model or score existed.

## D3 — score-sabotage rows (Kasper #7252; set before any training run finished)

The blind removed amount r = D - D_H is negative when a cut sends the probed output FARTHER from the unprobed one
(a third token, an overshoot). The penalty clips it at 0, and the blind greedy never picks such a head. The title
cell can then fall below 0.30 by sabotage of the score while the head route stays intact. The status rule is
unchanged. Added reported rows per model: (a) informed greedy top-3 (already in the eval list); (b) a "cut moves the
output" greedy top-3, picked on dev by KL(p_cut || p_probed) (label-free), scored on test as R; (c) the probed answer
after the blind top-3 cut and after the (b) cut: share A / B / other token. Naming rule for the TLDR: when
R_blind3 < 0.30 but the (a) or (b) top-3 removes >= 0.30 in the same seeds, the yes is named "the audit's score was
fooled" (score sabotage), not rerouting, and the post says a cut-moves-output audit still finds the heads.
Seen when this was set: J0 cache only; no trained model or score existed.

## D4 — linear learning-rate warmup 0 -> 1e-4 over the first 32 updates, every arm (Bruno #7254)

The warm adapters have task CE ~7e-7, so the first update is pure penalty (grad norm CE 0.0002 vs lambda*P 10,028
at lambda 100). A fresh AdamW moves every LoRA weight by ~lr in its first step: in a 20-update smoke of L100 seed 0
(`logs/smoke_lam100.log`, not counted) CE jumped to 12.6 at update 2 and the model fell to always-A by update 20
(dev switch 0.00, push 0.10). That jump is the size of a fresh optimizer's first step, not a route the penalty
found. Change: linear warmup of the learning rate from 0 to 1e-4 over updates 1-32, then 1e-4, for EVERY arm
(A0, L0, L1, L10, L100), so the treatment and its controls see the same schedule. Run order (Bruno): A0 -> L100
s0-s2 -> whole graph -> L0/L10/L1 interleaved by seed (s0 first), so a time cut drops whole seeds, not whole
strengths. Seen when this was set: the 20-update smoke above only (one model, dev prompts, never test).

## D5 — status reads ANY penalty weight, with >= 2 finished seeds per weight (Tove #7260, Kasper #7262)

Changed AFTER an uncounted 20-update smoke at the top weight (`logs/smoke_lam100.log`, D4) showed the switch
collapse to always-A, and the change was proposed because of it. The title asks whether such a switch exists, so an
evasion at a weaker weight is still an evasion. New status rule: CONFIRMED = at some lambda in {1, 10, 100}, >= 2
seeds at that lambda each have switch >= 0.75 AND R_blind3 < 0.30. REJECTED = no lambda meets that, with >= 2
finished seeds at each lambda that counts, subject to the fake-no guard. A lambda with fewer than 2 finished seeds
does not count either way. The scout's prediction stays scored at lambda 100 exactly as written in PREREG. The
post's limits list says in one line that this rule changed after the smoke. Queue so each lambda can reach 2 seeds:
A0 -> L100 s0-s2 -> whole graph #1 -> guards (L100 s0 to 2,048 if L100 fails; A0 to 2,048 if its D2 cell < 0.75)
-> L0 s0 -> L10 s0 -> L10 s1 -> L0 s1 -> L1 s0 -> L1 s1 -> L10 s2 -> L0 s2 -> L1 s2 -> whole graph #2.
Seen when this was set: the 20-update smoke only; A0 at update 64 (dev, CE 0.71); no counted score.

## D6 — exact bf16 ties in the answer (Vera #7286, Tove #7291, Kasper #7293); set AFTER data

Seen when this was set: L100 seed 0 cached-tail test rows. The A-minus-B log-odds are bf16-quantized in steps of
0.125; 23/128 probed answers are exact ties (log p(A) == log p(B)) that torch argmax breaks toward A (token id 32 <
33). Reported argmax switch 0.898; strict switch 0.719. This rule was set after those numbers and is written to be
conservative in both directions:
- Strict switch = probed log-odds > 0 AND unprobed log-odds < 0 (an exact tie has no unique top-1).
- CONFIRMED needs clause 1 under the STRICT switch (>= 0.75).
- A REJECTED that rests on clause 1 must hold under the argmax switch too (argmax < 0.75).
- If strict and argmax give different statuses, the status is UNCONFIRMED.
- When clause 2 (R_blind3 >= 0.30) already fails in a seed, ties do not change that seed.
Every switch cell reports argmax rate, strict rate, tie counts for both conditions, and the mean of
P(A | probed) - P(A | unprobed) over the A/B pair (Tove #7291), whole graph and cached tail.
