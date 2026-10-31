# DEVIATIONS from PREREG v1 (append only; each entry dated by git time and says whether data had been seen)

## D1 — before any TEST reply, judge label or score (git time is the record). Signed edit profile (Bruno #10938,
Kasper #10934 note 1, Vera #10940).
PREREG v1 defined the gate norm profile as n_t(m) = |m - a_t^gate| ||w||. It is now SIGNED: n_t(m) = (m - a_t^gate) ||w||,
for TWIN, D(c), RN, RD and FK alike. So each built or random edit is the gate's own per-token edit rotated onto a new
unit vector, and TWIN has exactly the gate's p and push at every token, also at generated tokens where a_t > m.
Vera's clean pass (TEST prompt tokens t >= 1): tokens with a_t > m are 0.14% at m = 6, 0.04% at m = 12, 0 at m >= 18;
p_TWIN/p_G under |.| vs signed = 0.9991 at m = 6 and 1.0000 at m >= 12, so p values barely move. Nothing else changes.

## D2 — AFTER gen1 was seen (base, G, G_on, G_off at 5 pins; TWIN_0), before any G_off_K or position row exists
(git time is the record). Robustness of "the rest of the gate's push misses the refusal direction" to how r is built
(Tove #10948, Kasper #10949, Bruno #10954). The STATUS rule is unchanged; these rows set only the post's wording.
(a) Clean-pass rows (no generation): cos(w, r) and cos(w_off, r) with w_off = w - (w . r_hat) r_hat, for DoM directions
    r (dev_harmful[:100] minus test_harmless[:100]) at the L13 output at each of the 9 post-instruction template tokens
    ['<|im_end|>','\n','<|im_start|>','assistant','\n','<think>','\n\n','</think>','\n\n'], and at the last token of
    the L12, L14, L15, L16 outputs. Random band = |cos| p99 of the 6144 L13 neurons vs the frozen r (0.084).
(b) Arm G_off_K: K = DoM r at the L13 output at each of the LAST 5 template tokens, then r at the L12 and L14 outputs
    at the last token; Gram-Schmidt in that order; a vector is kept only if its residual is >= 10% of its norm (k posted).
    G_off_K edit = (m - a_t)(w - Q Q^T w). Judged on TEST 200 at all 5 pins, same judge, runs after gen6, before ft7.
    Wording rule, fixed now: ratio K = EFFECT(G_off_K, 30) / EFFECT(G, 30).
      ratio K >= 0.5  -> the post may say the remainder acts through "a second direction" (beyond the refusal
                         directions we measured at this layer).
      ratio K < 0.25  -> the post says "our one-token refusal direction misses part of refusal", and the headline is
                         about the estimate of r, not about a second mechanism.
      in between      -> the post reports the number and claims neither.
(c) Reported next to the split: b on TEST halves (prompts 0-99 and 100-199) and TWIN-vs-gate gap at each pin with
    its 95% CI (same bootstrap as the STATUS rule).

## D3 — AFTER gen1 and Vera's downstream-layer numbers were seen (git time is the record). Title wording only
(Kasper #10973, Tove #10974, Felix #10975). PREREG v1 defines p at the layer-13 output, where the neuron writes, and
says the downstream row never rescues the title. The title now names that layer so a reader cannot answer it from a
later layer: "Does a neuron's push along the refusal direction, at the layer it writes to, predict how well pinning it
bypasses refusal?" Same hypothesis, same variable, same STATUS rule; no number changes.
D3 note (same minute, wording only): the post uses Selma's final form of the same title, "At the Layer Where It Writes,
Does a Neuron's Push Along the Refusal Direction Alone Predict How Strongly Pinning It Bypasses Refusal?" (#10978).
