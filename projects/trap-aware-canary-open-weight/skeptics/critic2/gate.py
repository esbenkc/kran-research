# Gate check with an own answer matcher: a dollar/number token equal to the manifest answer, and the stale value absent.
import json, re, sys, collections
P = sys.argv[1] + "/projects/trap-aware-canary-open-weight/web/out/"
def toks(s): return {re.sub(r"[,$]", "", t) for t in re.findall(r"\$?\d[\d,]*(?:\.\d+)?", str(s or ""))}
for ag in ["gpt-oss-20b-d4", "qwen3-8b"]:
    gets = collections.defaultdict(list)
    for line in open(P + ag + "/access.log"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5 and f[2] == "GET": gets[f[3].split("?")[0]].append(f[4])
    n = g = graw = dis = 0; ex = []
    for line in open(P + ag + "/manifest.jsonl"):
        m = json.loads(line)
        if m["cell"] != "neutral-none": continue
        rec = json.load(open(P + ag + "/" + m["raw_file"]))
        t = toks(rec.get("answer")); a = re.sub(r"[,$]", "", m["answer"]); s = re.sub(r"[,$]", "", m["stale"])
        mine = a in t and s not in t; fit = "200" in gets.get(m["fit_path"], [])
        n += 1; g += fit and mine; graw += fit and bool(rec["correct"])
        if mine != bool(rec["correct"]): dis += 1; ex.append((m["answer"], m["stale"], str(rec.get("answer"))[:80], rec["correct"]))
    print(ag, "UA n", n, "gate mine", g, "gate raw flag", graw, "disagree", dis); [print("   ", e) for e in ex[:6]]
