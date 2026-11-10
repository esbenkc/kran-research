# Deviations — trap-aware-canary

Each entry: what changed against PREREG v1 (frozen sections s0-s7), when, why, and which v1 rule stays a reported row.

## D1 (23:03 PST, before any counted run): block D runs after the cost cut if spend allows

s7 cut order removes block D first when the projection exceeds $4.50. The smoke projection ($4.88) triggered the cut
(D out; B and C to 30 per cell). After the cut the projection is $3.87. Block D (the `marked` minimal-pair side arm,
~$0.20) then runs last, after A, B and C, only if logged spend is <= $4.25. Reason: the $4.50 stop still holds, and D
is outside the prediction and the status. Decided from the smoke COST only; no smoke rate was read for it.
v1 rule as a reported row: if D does not run, the post says the minimal-pair arm was cut by the cost rule.
