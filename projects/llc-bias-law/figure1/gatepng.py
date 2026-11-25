"""Oracle Figure 1 gate on any PNG against the CURRENT draft: with-model-keys python3 gatepng.py <png> N
Copies the writer's draft into /tmp/f1llc/_notes (a scratch blog root: render.sh made assets/figures/<id>/ there)."""
import importlib.util
import shutil
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

loader = SourceFileLoader("oracle", str(Path.home() / "bin/oracle.py"))
spec = importlib.util.spec_from_loader("oracle", loader)
oracle = importlib.util.module_from_spec(spec)
sys.modules["oracle"] = oracle
loader.exec_module(oracle)

H = Path(__file__).resolve().parent
BLOG = H.parents[3] / "athanorscience.com"
S = Path("/tmp/f1llc")
(S / "_notes").mkdir(parents=True, exist_ok=True)
(S / "scripts").mkdir(exist_ok=True)
shutil.copy(BLOG / "scripts/figure-lint.mjs", S / "scripts/figure-lint.mjs")
post = S / "_notes/llc-bias-law.md"
shutil.copy(BLOG / "_notes/llc-bias-law.md", post)
png = Path(sys.argv[1]).resolve()
oracle.figure1_png = lambda d: png
n, ok, usd = int(sys.argv[2]), 0, 0.0
for k in range(n):
    rows, u = oracle.figure1_rows(post, "theory-1125-llc-bias-law")
    usd += u
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:900], flush=True)
print(f"{png.name}: {ok}/{n} PASS, ${usd:.4f}")
