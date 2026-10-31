# DEVIATIONS from PREREG v1 (append only; each entry dated by git time and says whether data had been seen)

## D1 — before any TEST reply, judge label or score (git time is the record). Signed edit profile (Bruno #10938,
Kasper #10934 note 1, Vera #10940).
PREREG v1 defined the gate norm profile as n_t(m) = |m - a_t^gate| ||w||. It is now SIGNED: n_t(m) = (m - a_t^gate) ||w||,
for TWIN, D(c), RN, RD and FK alike. So each built or random edit is the gate's own per-token edit rotated onto a new
unit vector, and TWIN has exactly the gate's p and push at every token, also at generated tokens where a_t > m.
Vera's clean pass (TEST prompt tokens t >= 1): tokens with a_t > m are 0.14% at m = 6, 0.04% at m = 12, 0 at m >= 18;
p_TWIN/p_G under |.| vs signed = 0.9991 at m = 6 and 1.0000 at m >= 12, so p values barely move. Nothing else changes.
