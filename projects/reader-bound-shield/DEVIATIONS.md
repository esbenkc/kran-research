# DEVIATIONS — reader-bound-shield

Every change to PREREG.md after the rules froze (v1 18780a90, v1.1 be6d6b2d, v1.2 ac76e137), with the time and what
had run when it was made.

## D1 — the plotted ceiling is the text difference D, not f (reporting only; 2026-11-09 17:11 PST)

Made after planted_s0 (the gate run) had started and before any gate read. Raised by Iris (#18643), conceded by
Kasper (#18645). PREREG v1.1 said the figure plots f (share of CRN pairs whose two thoughts differ). But y and x are
each a difference of two means of a [0,1] function of the thought text, so their text parts are bounded by the total
variation D between the probe-on and probe-off thought distributions, for any coupling. f >= D for every coupling
(coupling inequality), so f is the looser bound: the real probe's f 0.384 vs D 0.101 comes from the Gumbel pairing
(many tiny logit shifts flip the argmax at scattered positions), not from more signal.
Change: the figure's ceiling marker per arm = D of that arm's served read at the end of training (the read that gives
y; it bounds y), and the tooltip names the start D (it bounds x) and f. The post's main text uses D only; f goes to the
expert foldout. No rule, threshold or status input changes.

## D2 — the planted end-text reader can move ahead of slot 3 (2026-11-09 17:43 PST)

Made while the gate's first TEST look (update 100) was running and before its result existed. Raised by Iris
(#18659, #18661); decided by Felix. RL amplifies the planted cue (dev D 0.10 -> 0.27 at update 100), so the planted
switch y can pass its START reader edge (capped near start D 0.10) while staying under its END text difference. The
frozen status rule takes x = max(start x, end x) and its REJECTED clause fires when any RL point's y CI-low is above
its x CI-high by >= 1.5 points. Without the end-text fit, a missing cell would decide the status.
Rule: if the gate passes and planted y CI-low > planted start x CI-high (no margin; Kasper #18664: CONFIRMED needs no y CI-low above x CI-high, so +1.5 left a gap), the planted end-text read
(texts-only, TRAIN split, R 4, gate adapter) plus its reader fit (frozen rule, START adapter + dec-bias -2.255, Felix
#18660) moves ahead of slot 3. Otherwise the frozen order stands. Cost ~10 min read + 15-28 min fit.

## D3 — early stop of the gate at 200 when flat, and one dense-cue engine diagnostic (2026-11-09 18:14 PST)

Made BEFORE the update-200 look's result existed (look 1 at update 100: y -0.05 [-1.50, +1.41], forced switch +0.86,
D 0.261). Proposed by Tove (#18674); decided by Felix. Only the no-status branch changes (a pass at 200 keeps v1.2 + D2).
Rule: if look 200 does not pass AND its forced switch g(notNeed) - g(Need) < +3 points, the gate run stops at 200.
The post reports 'no switch by 200 updates; the 500-update clause was not run' (prereg outcome: unsettled on that
clause; the status is UNCONFIRMED either way, since real_s0 cannot fit after a late pass). If the forced switch is
>= +3 at 200, the frozen order to 500 stands (Felix #18668).
Freed lane, in order: (1) DENSE-CUE diagnostic: same engine, recipe, start adapter, seed 0 and dec-bias; the planted
' Need' cue with eps large enough that ' Need' is the first thought token on (nearly) every probed episode (D ~0.85
on the start policy, eps set offline from the start logits); 200 updates; one TEST read (R 4, forced gate) at the end.
Frozen bar: natural served y >= 10 points with CI-low > 0. Learns -> the engine can learn a switch and the planted
failure is signal density (~10% of probed rows carry the 10-point cue); does not learn -> the decision path is deaf.
Reported as an engine diagnostic, never a status input. (2) planted END-text read + reader fit (#18668 (d)). (3)
planted START reader fit. Dropped in this branch: the bit-exact read and the real START fit (Tove #18674).
