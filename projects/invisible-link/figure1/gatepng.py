"""Oracle Figure 1 gate on any PNG: with-model-keys python3 gatepng.py <png> N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
post = Path(__file__).resolve().parents[2] / "blog.kran.ai/_notes/invisible-link.md"
n, ok = int(sys.argv[2]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "defense-1108-invisible-link")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
