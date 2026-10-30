# DEVIATIONS — stack-conditional

Every change to the room goal or to PREREG.md after its first commit (5fd63e9). Each entry: what changed, who
decided (room post), and what had been seen when it was set. Times are Copenhagen time (git commit times).

## D0 (19:00, before any cache, smoke or RL data): test prompts are not new

The goal asked for "200 new prompts". The fixed 34-token prompt pool has exactly 646 Dolly-15k questions (Bruno
#9562). The 200 test prompts are therefore the held-out split of numeric-chameleon-rl: never trained on in this run,
but already used as that post's test set. A new 35-token pool (628 prompts) would change the compiled graph and the
base rates, so we kept the 34-token pool (Felix #9571; Tove #9567 and Kasper #9568 asked for the wording). Nothing
from any run in this room had been seen.

## D1 (19:03, after the 20-update smoke, before any main-arm run): readout and takeoff frozen in v1.2

The goal lists a logistic readout and a predicted takeoff among the numbers to freeze before training; v1.1 had
only the bit counts and T_pred. Iris (#9591) posted the takeoff after she had seen smoke steps 5-10 (P fell on both
builds together, gap 0.001). Both were added as PREREG section 8 before s0 and s1 started. The takeoff is scored
and reported; it is in no status or outcome rule.
