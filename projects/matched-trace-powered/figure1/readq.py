"""A/B the oracle Figure 1 gate (FIG1_READ + FIG1_MATCH, same cheap model) on any PNG(s), N parallel reads each.
    with-model-keys python3 figure1/readq.py N a.png [b.png ...]"""
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

ROOM = "oracle-1112-matched-trace"
post = Path(__file__).resolve().parents[4] / "blog.kran.ai/_notes/matched-trace-powered.md"
fm, body = oracle.front_matter(post.read_text())
tldr = re.search(r"^\W*TLDR\W*\s*(.*)$", body, re.M | re.I).group(1)
question = oracle.fm_val(fm, "question")


def one(png):
    r, m, _ = oracle.ask_cheap(oracle.FIG1_READ_PROMPT, ROOM, "gate:figure1", 120, calibrated_only=True, image=Path(png))
    q, a = str(r.get("question", "")), str(r.get("answer", ""))
    if not q or not a or q.upper().startswith("UNCLEAR") or a.upper().startswith("UNCLEAR"):
        return False, q, a, "UNCLEAR"
    v, _, _ = oracle.ask_cheap(oracle.fill(oracle.FIG1_MATCH_PROMPT, question=question, tldr=oracle.clip(tldr, 600), q=q, a=a),
                               ROOM, "gate:figure1", 120, calibrated_only=True)
    return bool(v.get("pass")), q, a, v.get("why", "")


n = int(sys.argv[1])
for png in sys.argv[2:]:
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, [png] * n))
    print(f"== {png}: {sum(r[0] for r in res)}/{n}")
    for ok, q, a, why in res:
        print(f"  {'PASS' if ok else 'FAIL'} Q: {q[:150]}\n       A: {a[:170]}" + ("" if ok else f"\n       -> {why[:200]}"))
