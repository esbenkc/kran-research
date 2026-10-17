# Hugo (skeptic 2) — sign-off on algzoo-432

Scope given by Magnus (#2094): the noise-floor bar definition and the bits accounting.
Own float64 numpy forward from `m16_10.pth` (torch max abs diff 0.0). Dev seeds 7000-7140 only;
holdout 432432 never used. I did not read or run the team's scripts; I read only the docstring
formula of `bits_curve.py` / `mixprec.py` to compare definitions.

Checkers: `skeptic2/hugo_check.py` (66 s) -> `hugo_check.json`; `skeptic2/hugo_noise_levels.py` (28 s)
-> `hugo_noise_levels.json`.

## Rebuilt numbers

| claim | team | mine (dev) | where |
|---|---|---|---|
| model accuracy | 95.3% | 0.95290 (seed 7001, 1e6) | hugo_check.py:43-46 |
| bar: 0.1% floor, 20 draws | 0.98671 (holdout), sd 0.0042 | 200 draws x 2e5: mean 0.98717 (SE 0.00026), median 0.98798, sd 0.0037; 20 x 1e6 on two noise seeds: 0.98862 / 0.98743 | hugo_check.py:50-66 |
| 0.03% floor | 0.99610 | 0.99623 (40 draws) | hugo_check.py:67-68 |
| **0.3% floor ("95.2%" in post)** | 0.9523 (Vera, 10 draws) | **0.9588 (200 draws, SE 0.0010)**; 10-draw means range 0.950-0.966 | hugo_noise_levels.py:22-28 |
| 374@4d original weights | 0.98933 holdout | 0.98942 (seed 7002, 1e6); 374 literals, W_hi in {-1,0,+1} | hugo_check.py:144-160 |
| fitted 364@3d | 0.99234 holdout | 0.99226 | same |
| fitted 340@3d | 0.98831 holdout | 0.98828 | same |
| mixprec 374 | 0.98835 holdout | 0.98838; digit hist 1/2/3/4 = 74/115/134/51 | same |
| **all 400 @3 digits ("98.9%" in post)** | 0.98858 | **0.98855 with output row 0 shifted to zero; 0.98421-0.98855 over the 10 equivalent row shifts, 7 of 10 below 0.98671** | hugo_check.py:91-95 |
| all 400 @2 digits ("about 70%") | ~0.70 | 0.623-0.761 over the 10 row shifts | same |
| all 400 @4 digits | — | 0.99826-0.99872 over the 10 row shifts | same |

## Bar definition

1. The prereg bar 0.98671 is an unbiased estimate. My 200-draw mean is 0.98717 +- 0.00026; the 20-draw
   procedure itself scatters by +-0.0008 (SE) and input seeds add about +-0.001. Holdout 0.98671 sits
   1.8 SE below my mean. No sign of a picked or tuned bar.
2. The draw distribution is skewed left (min 0.967, max 0.993, median 0.9880 > mean 0.9872). 37.5% of
   single noisy copies fall below the bar. A median definition would raise the bar by 0.08 pt. 374@4d
   (0.9894) and fitted 364@3d (0.9923) clear both; the "tie" rows stay ties.
3. The bar is calibrated at about 3 significant digits of the weights. Rounding the 400 gauge numbers to
   3 digits gives rms relative error 0.148%. Gaussian noise at that rms gives 0.9823; 0.1% noise in the
   same gauge gives 0.98681 (hugo_check.py:97-115). So "the network's own weights at 3 digits reach the bar" follows mostly from
   the choice of 0.1%, not from the network. The limits bullet on noise level covers this in part.
4. **3-digit rounding is gauge-dependent at the bar's own scale.** The W_oh row shift (footnote [^gauge])
   has 10 equal choices. With row 0 as reference, all-400@3d = 0.98855. With the other 9: 0.98667,
   0.98421, 0.98582, 0.98578, 0.98567, 0.98605, 0.98726, 0.98432, 0.98743. The post's "keeps 98.9%"
   is the best of the 10 bar one.

## Bits accounting

My own code (sign + mantissa log2(9*10^(d-1)) + empirical exponent entropy + log2 C(400,k) mask +
16 log2 3 for W_hi) agrees with the team's within 1% for every row with uniform digits:
374@4d 6,449 vs 6,458; 382@3d 5,282 vs 5,293; fitted 364@3d 5,084 vs 5,084; fitted 340@3d 4,823 vs 4,814;
all-400@3d 5,455 vs 5,479; mixprec 5,214 vs 5,208. Ranking unchanged.
Nit: `bits_curve.py:34` codes the W_oh mask as log2 C(150, k), but W_oh[1:] has 144 slots
(about +24 bits on the fitted 364 row). Does not change any conclusion.

## Novelty

Web search (2026-10-17): no newer work on M_16,10 than the ARC post, the Keane gist and the LessWrong
comments the post cites. The post claims "none found", not a record; nothing to beat.

## Fixes needed in `_notes/algzoo-432.md`

- F1 (line 63): "rounding every weight to 3 significant digits keeps 98.9% agreement" -> "keeps
  98.4% to 98.9%, depending on which of the equivalent forms we round".
- F2 (line 72): "only 95.2% of the time, like the 4-number rule" -> "about 96% of the time, close to
  the 4-number rule" (200 draws: 0.9588; the rule's 95.5% matches about 0.35% noise, 0.9505).

## Verdict

CONFIRMED as "none found" for the noise-floor bar and the bits ledger, after F1 and F2.
