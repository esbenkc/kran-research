"""Oracle Figure 1 gate on any PNG: with-model-keys python3 gatepng.py <png> N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
post = Path(__file__).resolve().parents[3] / "blog.kran.ai/_notes/rope-sink-band.md"
n, ok = int(sys.argv[2]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "theory-1102-rope-sink-band")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:260], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
