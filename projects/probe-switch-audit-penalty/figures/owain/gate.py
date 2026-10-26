# Local replica of oracle.py's Figure 1 cold-reader gate (FIG1_READ_PROMPT + FIG1_MATCH_PROMPT, haiku). Usage: gate.py png n
import base64, json, re, sys, anthropic
sys.path.insert(0, "/Users/esben/bin")
src = open("/Users/esben/bin/oracle.py").read()
READ = re.search(r'FIG1_READ_PROMPT = """(.*?)"""', src, re.S).group(1)
MATCH = re.search(r'FIG1_MATCH_PROMPT = """(.*?)"""', src, re.S).group(1).replace("{{", "{").replace("}}", "}")
post = open("/Users/esben/work/oracle-1026-audit-penalty/blog.kran.ai/_notes/probe-switch-audit-penalty.md").read()
title = re.search(r"^title:\s*(.*)$", post, re.M).group(1).strip()
tldr = re.search(r"^\W*TLDR\W*\s*(.*)$", post, re.M | re.I).group(1)[:600]
png, n = sys.argv[1], int(sys.argv[2])
img = base64.b64encode(open(png, "rb").read()).decode()
a = anthropic.Anthropic()
def ask(content):
    r = a.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=[{"role": "user", "content": content}])
    t = r.content[0].text
    return json.loads(t[t.index("{"):t.rindex("}") + 1])
ok = 0
for i in range(n):
    r = ask([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}, {"type": "text", "text": READ}])
    v = ask(MATCH.replace("{title}", title).replace("{tldr}", tldr).replace("{q}", r["question"]).replace("{a}", r["answer"]))
    ok += bool(v["pass"])
    print(("PASS" if v["pass"] else "FAIL"), "| Q:", r["question"], "| A:", r["answer"], "|", "" if v["pass"] else v["why"])
print(f"{ok}/{n}")
