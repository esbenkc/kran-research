"""Replay oracle.py's Figure 1 gate n times (Owain). Usage: python3 fig1gate.py <post.md> [n]"""
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

loader = SourceFileLoader("oracle", str(Path.home() / "bin/oracle.py"))
spec = importlib.util.spec_from_loader("oracle", loader)
o = importlib.util.module_from_spec(spec)
sys.modules["oracle"] = o
loader.exec_module(o)

post, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3
ok, usd = 0, 0.0
for i in range(n):
    rows, u = o.figure1_rows(post, "theory-1125-llc-bias-law")
    usd += u
    for st, name, det in rows:
        if "cold reader" in name or st != "PASS":
            print(f"{i + 1}: {st} {name} | {det}")
            ok += st == "PASS" and "cold reader" in name
print(f"gate {ok}/{n}, ${usd:.3f}")
