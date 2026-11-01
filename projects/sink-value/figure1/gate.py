"""Run oracle's Figure 1 cold-reader gate N times on the blog clone's figure1.png.
    with-model-keys python3 gate.py N
Uses Selma's current _notes/sink-value.md; while its figure1_alt is still a SLOT, a temp copy (in /tmp, with
assets/ and scripts/ symlinked to the blog clone) gets the alt text from alt.txt. The post file is never written."""
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

H = Path(__file__).resolve()
post = next(p for p in [H.parents[2] / "blog.kran.ai/_notes/sink-value.md", H.parents[4] / "blog.kran.ai/_notes/sink-value.md"]
            if p.exists())
blog = post.parents[1]
text = post.read_text()
if re.search(r'^figure1_alt:\s*"?SLOT', text, re.M):
    alt = (H.parent / "alt.txt").read_text().strip().replace('"', "'")
    text = re.sub(r"^figure1_alt:.*$", lambda m: f'figure1_alt: "{alt}"', text, count=1, flags=re.M)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "_notes").mkdir()
    for d in ("assets", "scripts"):
        (tmp / d).symlink_to(blog / d)
    post = tmp / "_notes/sink-value.md"
    post.write_text(text)
n, ok = int(sys.argv[1]), 0
for k in range(n):
    rows, usd = oracle.figure1_rows(post, "oracle-1101-sink-value")
    r = rows[-1]
    ok += r[0] == "PASS"
    print(k, r[0], r[2][:500], f"${usd:.4f}", flush=True)
print(f"{ok}/{n} PASS")
