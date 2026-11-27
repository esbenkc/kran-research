"""Oracle Figure 1 gate on any PNG against a post: with-model-keys python3 gatepng.py <png> <post.md> N"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

png = Path(sys.argv[1]).resolve()
post = Path(sys.argv[2]).resolve()
oracle.figure1_png = lambda d: png
n, ok = int(sys.argv[3]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "theory-1127-tell-chain")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
