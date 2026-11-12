"""Oracle Figure 1 gate on any PNG: with-model-keys python3 gatepng.py <png> N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
post = next(d for d in Path(__file__).resolve().parents if (d / "blog.kran.ai").is_dir()) / "blog.kran.ai/_notes/optimizer-cue-size.md"
n, ok = int(sys.argv[2]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "theory-1112-optimizer-sets")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
