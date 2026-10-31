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

## D4 — AFTER D2 was seen (G_off_K ratio 0.455 [0.35, 0.56]; w_off cos -0.212 with the DoM direction at template
position -8), before any G_off_K9 reply exists (git time is the record). Wording rows only; the STATUS rule is
unchanged (Kasper #11010, Tove #11011, Iris #11005).
Gap in D2: K covered the last 5 template tokens (positions -5..-1) and the L12/L14 last token, but not position -8 (the
newline right after <|im_end|>), where neuron 3270 fires hardest on TEST harmful (mean clean activation -9.53) and where
w_off has its largest cosine with a refusal direction (-0.212).
Arm G_off_K9: K9 = DoM r (dev_harmful[:100] minus test_harmless[:100]) at the L13 output at ALL 9 template tokens, then
r at the L12 and L14 outputs at the last token; Gram-Schmidt in that order; keep a vector if its residual is >= 10% of
its norm (k posted). Edit = (m - a_t)(w - Q Q^T w). Judged on TEST 200 at pins 18, 24, 30 (pin 30 decides), same judge,
right after ft7. Ratio K9 = EFFECT(G_off_K9, 30) / EFFECT(G, 30), 95% CI by the frozen paired bootstrap.
Wording rule, fixed now:
  ratio K9 >= 0.5  -> the post may say "about half comes from outside every refusal direction we measured".
  ratio K9 < 0.25  -> the post says the rest acts along the refusal direction at the token where the neuron fires, and
                      that a last-token direction misses it; that becomes the headline. The status stays REJECTED for
                      the frozen title, because the frozen p uses the last-token r.
  in between       -> the post reports the number and claims neither.
Until G_off_K9 lands, the post names K as "seven refusal directions from the last five template tokens and two nearby
layers", never "every refusal direction we measured".

## D5 — disclosed after all judged cells were labelled (git time is the record). Judge temperature (Bruno #11025,
Kasper #11029). PREREG v1 said the judge runs at temperature 0. The API rejects a temperature parameter for
claude-haiku-4-5 (HTTP 400 "temperature: Extra inputs are not permitted", tested 05:35 PDT), so every label was sampled
at the model default (as in the last room's judge). Size of the effect, measured: a second, uncached labelling of the 5
headline cells (base, G@30, G_on@30, G_off@30, G_off_K@30; 1,000 labels, retest.py -> data/judge_retest/summary.json,
0.28 USD) agreed with the first on 96.5-98.5% of replies per cell and gave G +56.5, G_on +18.0, G_off +37.0, G_off/G
0.65 (first pass 0.63). Vera's judge with a different template gave G_off/G 0.60 [0.50, 0.71]. No status changes.
D4 note (after its run): Bruno ran G_off_K9 at all 5 pins (a superset of D4's 18/24/30) and before the rest of ft7
rather than after it (#11036). K9 kept all 11 vectors (9 template tokens at L13 + L12, L14 last token). Ratio K9 at
pin 30 = 0.375 [0.27, 0.47] -> the in-between band: the post reports the number and claims neither wording.
