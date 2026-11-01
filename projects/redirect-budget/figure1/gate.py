"""Run oracle's Figure 1 cold-reader gate N times. with-model-keys python3 gate.py POST ROOM N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
post, room, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if len(sys.argv) > 4:  # judge another PNG (e.g. the old figure) against the same post
    oracle.figure1_png = lambda d, _p=Path(sys.argv[4]): _p
ok = 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, room)
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[1], r[2][:300], ("WHY " + r[2].split(" -> ", 1)[1]) if " -> " in r[2] else "", f"${usd:.4f}", flush=True)
print(f"{ok}/{n} PASS")
