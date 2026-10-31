"""Run oracle's real Figure 1 gate N times on the post. with-model-keys python3 gate.py N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin")); import oracle  # noqa: E402
post = next(q / "blog.kran.ai/_notes/honeypot-talk.md" for q in Path(__file__).resolve().parents
            if (q / "blog.kran.ai/_notes/honeypot-talk.md").exists())
ok = 0
for i in range(int(sys.argv[1])):
    rows, _ = oracle.figure1_rows(post, "defense-1031-say-do-trap-gap")
    r = [x for x in rows if "cold reader" in x[1]][0]
    ok += r[0] == "PASS"
    print(r[0], str(r[2])[:600], flush=True)
print(f"{ok}/{sys.argv[1]} PASS")
