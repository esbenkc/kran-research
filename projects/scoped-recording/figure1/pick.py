"""List candidate Figure 1 examples: same agent + seed across the five arms, route taken and the thought after the
first denial. Run from the work dir: python3 swarm-defense-1107-scoped-recording/owain/pick.py [raw|raw_smoke] [filter]
filter: cross  = key arm flag-only AND flag arm key-only (crossover in one agent+seed)
        keyflag = key arm flag-only (any other arm)
        all    = everything
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "kran-research/projects/scoped-recording"
sys.path.insert(0, str(P))
from scorer import score  # noqa: E402

RAW = P / (sys.argv[1] if len(sys.argv) > 1 else "raw")
WANT = sys.argv[2] if len(sys.argv) > 2 else "cross"
ARMS = ["none", "all", "key", "flag", "keyonly"]


def route(s):
    if s["flag"] and s["key"]:
        return "BOTH"
    return "flag" if s["flag"] else "key" if s["key"] else "-"


def first_thought(rec):
    """Thought of the reply right after the first observation that shows the record."""
    obs = rec["observations"]
    i = next((k for k, o in enumerate(obs) if "last archive record" in str(o)), None)
    if i is None or i >= len(rec["model_replies"]):
        return None, None
    t = rec["model_replies"][i]["text"]
    try:
        o = json.loads(t[t.index("{"): t.rindex("}") + 1])
        return o.get("thought"), o.get("tool") + " " + json.dumps(o.get("args"))[:120]
    except Exception:
        return t[:200], None


runs = {}
for p in sorted(RAW.glob("*/*/*.jsonl")):
    rec = json.loads(p.read_text())
    runs[(rec["model"], rec["cell"], rec["seed"])] = rec

keys = sorted({(m, s) for m, c, s in runs})
for m, s in keys:
    if not all((m, a, s) in runs for a in ARMS):
        continue
    sc = {a: score(runs[(m, a, s)]) for a in ARMS}
    r = {a: route(sc[a]) for a in ARMS}
    ok = {"cross": r["key"] == "flag" and r["flag"] == "key",
          "keyflag": r["key"] == "flag",
          "all": True}[WANT]
    if not ok:
        continue
    print(f"== {m} seed {s}  " + "  ".join(f"{a}:{r[a]}" for a in ARMS))
    for a in ARMS:
        t, act = first_thought(runs[(m, a, s)])
        print(f"   {a:8s} {r[a]:5s} | {(t or '')[:230]}\n            -> {act}")
