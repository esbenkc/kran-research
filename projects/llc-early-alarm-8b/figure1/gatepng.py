"""Oracle Figure 1 gate on any PNG against the CURRENT draft: with-model-keys python3 gatepng.py <png> N
Copies the writer's draft into /tmp/f1b/_notes (a scratch blog root: render.sh made assets/figures/<id>/ there)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

H = Path(__file__).resolve().parent
S = Path("/tmp/f1b")
(S / "_notes").mkdir(parents=True, exist_ok=True)
(S / "scripts").mkdir(exist_ok=True)
shutil.copy(H.parents[1] / "blog.kran.ai/scripts/figure-lint.mjs", S / "scripts/figure-lint.mjs")
post = S / "_notes/llc-early-alarm-8b.md"
shutil.copy(H.parents[1] / "blog.kran.ai/_notes/llc-early-alarm-8b.md", post)
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
n, ok = int(sys.argv[2]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "theory-1124-llc-early-alarm-8b")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS")
