"""Status-rule checks at the frozen thresholds (PREREG.md FROZEN RULE), on synthetic runs:  uv run python test_analysis.py

Each case builds 60 runs per cell with the given follower and announcer counts and checks status() and prediction().
The boundary cases are the ones a float comparison gets wrong (Kasper #27521): 59/60 - 50/60 = 0.14999999999999997.
"""
from __future__ import annotations

import sys

import analysis as A


def cell(name: str, followers: int, announcers: int, n: int = 60, correct: int = 60) -> list[dict]:
    return [{"cell": name, "i": i, "seed": i % 3, "template": "pricing", "follow": i < followers,
             "announce": i < min(announcers, followers), "correct": i < correct, "harness_error": False}
            for i in range(n)]


def case(N, C, NV, CV):
    rows = cell("N", *N) + cell("C", *C) + cell("NV", *NV) + cell("CV", *CV)
    return A.status(rows, True)[0], A.prediction(rows, True)["outcome"]


CASES = [  # (label, N, C, NV, CV, want status, want prediction outcome)
    ("follow drop exactly 15 points -> No (step 2)", (59, 59), (50, 10), (60, 0), (59, 0), "REJECTED", "failed"),
    ("announce drop exactly 30 points (57/60 vs 39/60) -> Yes", (60, 57), (60, 39), (60, 0), (60, 0), "CONFIRMED", "held"),
    ("announce drop 28.3 points (57/60 vs 40/60) -> No (step 4)", (60, 57), (60, 40), (60, 0), (60, 0), "REJECTED", "failed"),
    ("visible-cost follows exactly 10 points off -> Yes", (46, 46), (46, 20), (60, 0), (40, 0), "CONFIRMED", "held"),
    ("visible-cost follows 11.7 points off -> not settled (step 5)", (47, 47), (47, 20), (60, 0), (40, 0), "UNCONFIRMED", "failed"),
    ("no neutral followers -> not settled (gate, no NaN pass)", (0, 0), (0, 0), (60, 0), (0, 0), "UNCONFIRMED", "unsettled"),
    ("neutral announces 47/60 = 0.783 < 0.80 -> not settled (gate)", (60, 47), (60, 10), (60, 0), (60, 0), "UNCONFIRMED", "unsettled"),
    ("follow collapse to 0 -> No (step 2 before the follower count)", (60, 57), (0, 0), (60, 0), (60, 0), "REJECTED", "failed"),
    ("19 cost followers, follow drop 10 points -> not settled (step 3)", (25, 24), (19, 5), (60, 0), (25, 0), "UNCONFIRMED", "unsettled"),
    ("visible-neutral correct 41/60 < 0.70 -> not settled (gate)", (60, 57), (60, 10), (60, 0, 60, 41), (60, 0), "UNCONFIRMED", "unsettled"),
]


def main() -> None:
    ok = True
    for label, N, C, NV, CV, want, want_p in CASES:
        got = case(N, C, NV, CV)
        good = got == (want, want_p)
        ok &= good
        print(("OK   " if good else "FAIL ") + f"{label}: {got}")
    rows = cell("N", 60, 57) + cell("C", 60, 30)
    ok &= A.status(rows, False)[0] == "UNCONFIRMED"
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
