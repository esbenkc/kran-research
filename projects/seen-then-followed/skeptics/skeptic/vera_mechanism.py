"""Check Iris #25975 / Tove mechanism facts from raw transcripts (own code)."""
import sys, json, collections
sys.argv = ['x', '../../kran-research/projects/seen-then-followed', 'main']
exec(open('vera_check.py').read().split('if __name__')[0])
runs = load('main'); cells = by_cell(runs); C = counted(cells, prefix_k(cells))
def tpl(x): return json.load(open(x["f"]))["site"]["template"]
L = C[("lcpw", B)]
miss = [x for x in L if x["reach"] and not x["follow"]]
both = sum({"extract_text", "extract_hyperlinks"} <= set(x["tools_called"]) for x in miss)
print(f"lcpw 26B misses {len(miss)}; called both extract_text and extract_hyperlinks: {both}")
L31 = C[("lcpw", A)]
print("lcpw 31B runs calling both:", sum({"extract_text", "extract_hyperlinks"} <= set(x["tools_called"]) for x in L31), "/", len(L31))
for c in [("lcpw", B), ("lcpw", A), ("mcpfetch", B), ("mcpfetch", A)]:
    t = collections.defaultdict(lambda: [0, 0, 0])
    for x in C[c]:
        k = tpl(x); t[k][0] += x["follow"]; t[k][1] += x["reach"]; t[k][2] += 1
    print(c, "template: follow/reach/n", {k: tuple(v) for k, v in sorted(t.items())})
ans = collections.Counter(); steps = collections.Counter(); corr = 0
for x in miss:
    d = json.load(open(x["f"]))
    corr += bool(d.get("correct"))
    if (d["actions"] or [{}])[-1].get("tool") == "answer":
        steps[len(d["actions"])] += 1
        ans[(d["site"]["template"], (d.get("answer") or "")[:70], d["site"]["answer"])] += 1
print("misses correct:", corr, "| answered misses by number of actions:", dict(steps))
for k, v in ans.most_common(20): print("  ", v, k)
