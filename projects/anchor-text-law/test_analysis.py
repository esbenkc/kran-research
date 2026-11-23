"""Boundary tests of the frozen rule in analysis.py on synthetic rows (no model, no data):  uv run python test_analysis.py

Base case (exact boundaries, all must PASS with Fractions; floats would wobble on 1 - 0.15 and 0.9 - 0.8):
  A = 60 runs/cell, B = 42, every run reached.
  A raw 54/60 follow -> follow_A = 9/10; B md 42/42, B anchor 42/42 -> follow_B = 1.
  A md 60/60 -> err = 1 - 1 x 9/10 = +1/10 exactly (passes, inclusive).
  A bare 51/60 = 85/100 = fr(md, A) - 15/100 exactly -> G fires for A (inclusive).
  A anchor 54/60 -> fr(anchor) - fr(md) = -1/10 exactly (passes); err = 9/10 - 9/10 = 0.
  B raw 42/42 -> err 0; B bare 42/42 (no G for B).  => CONFIRMED.
"""
from __future__ import annotations

import sys
from fractions import Fraction as F

import analysis as an

ok_all = True


def check(ok: bool, msg: str) -> None:
    global ok_all
    ok_all &= bool(ok)
    print(("OK   " if ok else "FAIL ") + msg)


def rows(n: int, follows: int, reached: int | None = None, m="A") -> list[dict]:
    reached = n if reached is None else reached
    return [{"reach": i < reached, "follow": i < follows, "block": i % 3, "model": m} for i in range(n)]


def base() -> dict:
    return {("A", "raw"): rows(60, 54), ("A", "md"): rows(60, 60), ("A", "bare"): rows(60, 51), ("A", "anchor"): rows(60, 54),
            ("B", "raw"): rows(42, 42), ("B", "md"): rows(42, 42), ("B", "bare"): rows(42, 42), ("B", "anchor"): rows(42, 42)}


GATE = {"A": (30, 30), "B": (30, 30)}
FLOOR = {"A": 60, "B": 42}


def run(cell, gate=GATE, harness=None, floor=FLOOR, draws=400):
    harness = harness or {k: (0, len(v)) for k, v in cell.items()}
    return an.status(cell, gate, harness, floor, an.bootstrap(cell, draws=draws))


def main() -> None:
    c = base()
    lw = an.law(c)
    check(lw["A", "md"]["err"] == F(1, 10), f"err(md, A) = +1/10 exactly ({lw['A', 'md']['err']})")
    check(float(1) - 0.9 != 0.1, "the float version of that err is not 0.1 (why Fractions)")
    s = run(c)
    check(s["status"] == "CONFIRMED" and s["G_fired"] == {"A": True, "B": False}, f"base case CONFIRMED, G fires for A only: {s['status']} {s['why']}")
    check(s["prediction"]["C1"] and s["prediction"]["C3"] and not s["prediction"]["C2"], f"prediction C1, C3 hold, C2 fails (err bare B 0): {s['prediction']}")

    c = base(); c["A", "bare"] = rows(60, 52)  # 52/60 > 85/100: no G anywhere
    s = run(c)
    check(s["status"] == "UNCONFIRMED" and s["why"] == "no G", f"bare 52/60: no G -> UNCONFIRMED ({s['why']})")

    c = base(); c["A", "anchor"] = rows(60, 53)  # anchor - md = 53/60 - 1 = -7/60 < -1/10: anchor clause fails
    s = run(c)
    check(s["status"] != "CONFIRMED", f"anchor 53/60 (gap -7/60): not CONFIRMED ({s['status']})")

    c = base(); c["A", "md"] = rows(60, 60); c["A", "raw"] = rows(60, 53)  # err(md,A) = 1 - 53/60 = 7/60 > 1/10
    s = run(c)
    check(s["status"] != "CONFIRMED", f"err 7/60 in a status cell: not CONFIRMED ({s['status']})")

    c = base(); c["B", "raw"] = rows(42, 20)  # err(raw, B) = 20/42 - 1: interval wholly below -1/10
    s = run(c)
    check(s["status"] == "REJECTED" and s["rejected_a"], f"raw B 20/42: REJECTED (a) ({s['why']})")

    c = base(); c["A", "anchor"] = rows(60, 30); c["A", "md"] = rows(60, 60); c["A", "bare"] = rows(60, 30)
    s = run(c)
    check(s["status"] == "REJECTED" and s["rejected_b"], f"G for A and anchor 30/60 vs md 60/60: REJECTED (b) ({s['why']})")

    c = base(); c["B", "raw"] = rows(42, 20)
    s = run(c, gate={"A": (30, 30), "B": (20, 30)})
    check(s["status"] == "UNCONFIRMED" and s["why"] == "a gate failed", "skill gate 20/30 beats a REJECTED pattern -> UNCONFIRMED")
    s = run(base(), gate={"A": (21, 30), "B": (21, 30)})
    check(s["status"] == "CONFIRMED", "skill gate 21/30 passes (inclusive)")
    s = run(base(), floor={"A": 60, "B": 39})
    check(s["status"] == "UNCONFIRMED", "floor 39 < 42 -> UNCONFIRMED")
    h = {k: (0, len(v)) for k, v in base().items()}; h["A", "raw"] = (2, 60)  # 2/60 > 2/100
    check(run(base(), harness=h)["status"] == "UNCONFIRMED", "harness errors 2/60 > 2% -> UNCONFIRMED")
    h["A", "raw"] = (1, 60)
    check(run(base(), harness=h)["status"] == "CONFIRMED", "harness errors 1/60 <= 2% passes")

    c = base(); c["A", "raw"] = rows(60, 19, reached=19)  # follow_A rests on 19 reached runs < 20: law undecidable
    s = run(c)
    check(s["status"] == "UNCONFIRMED" and not s["law_decidable"], f"19 reached runs in A raw: law undecidable -> UNCONFIRMED ({s['why']})")

    # percentile: numpy linear interpolation on exact values
    xs = [F(k, 7) for k in range(11)]
    check(an.pct(xs, F(1, 2)) == F(5, 7) and an.pct(xs, F(25, 1000)) == F(1, 4) * F(1, 7), "pct interpolates like numpy, exactly")
    # seed level: 3 block values
    sl = an.seed_level(base())
    check([x for x in sl["err", ("A", "md")]["blocks"]] == [F(1, 10)] * 3, f"seed-level err blocks A md = 1/10 x 3 ({sl['err', ('A', 'md')]['blocks']})")
    print("ALL OK" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
