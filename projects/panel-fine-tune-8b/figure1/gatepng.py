"""Oracle Figure 1 gate on any PNG: with-model-keys python3 gatepng.py <png> N <blog clone>/_notes/panel-fine-tune-8b.md"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
post = Path(sys.argv[3]).resolve()
n, ok = int(sys.argv[2]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "oracle-1116-panel-fine-tune-8b")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
