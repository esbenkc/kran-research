"""Run oracle's Figure 1 cold-reader gate N times on the post (read only).  with-model-keys python3 gate.py N"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

post = Path(__file__).resolve().parents[4] / "blog.kran.ai/_notes/matched-trace-powered.md"
n, ok = int(sys.argv[1]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "oracle-1112-matched-trace")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2], f"${usd:.4f}", flush=True)
print(f"{ok}/{n} PASS")
