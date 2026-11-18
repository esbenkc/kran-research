"""Run plain-English lint, post_lint and N cold-read gates on a post (optionally mock-filled)."""
import sys, re, importlib.util, concurrent.futures as cf
spec = importlib.util.spec_from_file_location("oracle", "/Users/esben/bin/oracle.py")
o = importlib.util.module_from_spec(spec); sys.modules["oracle"] = o; spec.loader.exec_module(o)
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
md = open(path).read()
mock = {k: v for k, v in (a.split("=", 1) for a in sys.argv[3:])}
if "--mock" in sys.argv or True:
    md2 = re.sub(r"\[FILL[^\]]*\]", lambda m: "", md)
else:
    md2 = md
for line, prob, sent in o.plain_findings(md2):
    print(f"PLAIN L{line}: {prob}: {sent[:140]}")
for ok, name, det in o.post_lint(md2):
    if not ok: print(f"LINT FAIL {name}: {det[:200]}")
if n:
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda _: o.cold_read_gate(md2), range(n)))
    for ok, det, model, usd in res:
        print("COLD", "PASS" if ok else "FAIL", f"${usd:.3f}", det.replace("\n", " ")[:700])
