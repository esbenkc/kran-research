# Skeptic 3 (Nadia) Sign-off

**Status:** REJECTED (NO)

## Requirements Checklist
- [x] **Own Derivation & Equations Check:** I wrote a short derivation in `derivation.md` covering $N(ch)=N(h)$, Euler $\nabla_h \text{KL} \cdot h = 0$ and $H h = 0$ at the clean state, and the $R_0$ gap-weighted pair formula. I implemented a symbolic and numeric check in `sympy_check.py` verifying these properties. I confirm that the post's displayed equations match my independent derivation exactly.
- [x] **Own Code from Raw States:** `check.py` independently computes gap drop, restore share (Title Metric $R$), and catch rates. Threshold rule strictly uses the 2nd-highest of 200 line-set true statements as defined in PREREG.
- [x] **Novelty Check:** Included in `novelty.md`. Novel mechanistic insight that the optimizer ignores the exact free symmetry.
- [x] **Null Tests:**
  - **Random Direction Radial Energy Share (Null Test 1):** 0.071% (Theoretical $\approx 1/1536$).
  - **Natural True-vs-False Radial Energy Share (Null Test 2):** 7.78%.
  - **Post-norm arm B (r29):** Tested for seeds 0 and 1. Yields $R \approx -0.015$ to $-0.019$, showing length restores don't contribute meaningfully as expected.

## Title Metric Verification
- Seed 0 (arm A): Gap drop 99.84%, R = 0.01005
- Seed 1 (arm A): Gap drop 98.46%, R = 0.01429
- Seed 2 (arm A): Gap drop 98.61%, R = 0.00845

**Verdict**: The precondition (gap drop > 50%) is comfortably met for all three seeds (> 98%), and all three seeds have $R < 0.5$ (in fact $R \approx 0.01$). Thus, gradient training does **not** find the exact free layer-norm rescale but instead relies on tangential moves. The title claim is strictly **REJECTED (NO)**.
