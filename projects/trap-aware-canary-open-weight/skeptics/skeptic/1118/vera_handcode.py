# Vera: independent scorer for the blind hand code. Builds its OWN key from the fit ids quoted in packets.md
# (12-hex ids -> manifest hid), never from handcode_key.json; then compares with the committed key if present.
# Usage: python3 vera_handcode.py <project dir>
import json, sys, re, csv, random, collections, os, hashlib

P = sys.argv[1]
man = [json.loads(l) for l in open(f"{P}/web/out/gpt-oss-20b-d4/manifest.jsonl")]
by_hid = {m["hid"]: m for m in man}
CELL = {"warning-cued": "W", "neutral-cued": "U"}
txt = open(f"{P}/handcode/packets.md").read()
items = re.split(r"^## (H\d{3})\s*$", txt, flags=re.M)
mykey = {}
for name, body in zip(items[1::2], items[2::2]):
    hids = {h for h in re.findall(r"[0-9a-f]{12}", body) if h in by_hid}
    cells = {(by_hid[h]["cell"], by_hid[h]["i"]) for h in hids}
    mykey[name] = sorted(cells)
    mykey[name] = sorted(cells)
    if len(cells) != 1:  # fallback: match the packet's reasoning blocks to raw reasoning of W/U runs
        blocks = [b.strip() for b in re.split(r"\[step \d+ (?:reasoning|thought)\]", body.split("```text", 1)[-1].rsplit("```", 1)[0]) if len(b.strip()) > 40]
        hits = set()
        for m in man:
            if m["cell"] not in CELL:
                continue
            r = json.load(open(f"{P}/web/out/gpt-oss-20b-d4/raw/{m['cell']}/{m['i']:03d}.json"))
            alltxt = "\n".join((x.get("reasoning") or "") + "\n" + (x.get("text") or "") for x in r["replies"])
            if blocks and all(b[:120] in alltxt for b in blocks):
                hits.add((m["cell"], m["i"]))
        mykey[name] = sorted(hits)
amb = {k: v for k, v in mykey.items() if len(v) != 1}
print("items", len(mykey), "uniquely mapped", len(mykey) - len(amb), "unmapped/ambiguous", {k: v for k, v in list(amb.items())[:5]})
key = {k: v[0] for k, v in mykey.items() if len(v) == 1}
print("mapped cells", collections.Counter(CELL.get(c, c) for c, _ in key.values()))
kp = f"{P}/handcode/handcode_key.json"
if os.path.exists(kp):
    raw = open(kp, "rb").read()
    print("key sha256", hashlib.sha256(raw).hexdigest())
    k2 = json.loads(raw)
    print("committed key sample", list(k2.items())[:2] if isinstance(k2, dict) else k2[:2])

coders = {}
for f in sorted(os.listdir(f"{P}/handcode")):
    if f.startswith("coder_") and f.endswith(".csv") and "TEMPLATE" not in f:
        coders[f[6:-4]] = {r["item"]: r for r in csv.DictReader(open(f"{P}/handcode/{f}"))}
print("coders", {c: len(v) for c, v in coders.items()})


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n; pb = sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


names = sorted(key)
cs = list(coders)
for lab in "HTA":
    vals = {c: [int(coders[c][n][lab]) for n in names] for c in cs}
    line = f"{lab}:"
    for c in cs:
        for cell in ("W", "U"):
            sel = [v for n, v in zip(names, vals[c]) if CELL[key[n][0]] == cell]
            line += f" {c}-{cell} {sum(sel)}/{len(sel)}"
    if len(cs) == 2:
        a, b = vals[cs[0]], vals[cs[1]]
        line += f" | kappa {kappa(a, b):.3f} agree {sum(x == y for x, y in zip(a, b))}/{len(a)}"
        for mode, f in (("both", min), ("either", max)):
            comb = [f(x, y) for x, y in zip(a, b)]
            w = [v for n, v in zip(names, comb) if CELL[key[n][0]] == "W"]
            u = [v for n, v in zip(names, comb) if CELL[key[n][0]] == "U"]
            rng = random.Random(1); diffs = []
            for _ in range(10000):
                ws = [rng.choice(w) for _ in w]; us = [rng.choice(u) for _ in u]
                diffs.append(sum(ws) / len(ws) - sum(us) / len(us))
            diffs.sort()
            line += f" | {mode} W {sum(w)}/{len(w)} U {sum(u)}/{len(u)} W-U {sum(w)/len(w)-sum(u)/len(u):+.3f} [{diffs[250]:+.3f},{diffs[9749]:+.3f}]"
    print(line)

# Post-coding comparison (Iris #24175): Iris note tag 'contrast' vs Kasper A.
if "research" in coders and "critic" in coders:
    con = [int("contrast" in (coders["research"][n].get("note") or "").lower()) for n in names]
    ka = [int(coders["critic"][n]["A"]) for n in names]
    w = lambda v: sum(x for n, x in zip(names, v) if CELL[key[n][0]] == "W")
    u = lambda v: sum(x for n, x in zip(names, v) if CELL[key[n][0]] == "U")
    both = [min(a, b) for a, b in zip(con, ka)]
    print(f"contrast(Iris) W {w(con)}/60 U {u(con)}/60 | A(Kasper) W {w(ka)}/60 U {u(ka)}/60 | kappa {kappa(con, ka):.3f} agree {sum(a == b for a, b in zip(con, ka))}/120 | both W {w(both)} U {u(both)}")
    rng = random.Random(2)
    for lab, v in (("contrast", con), ("kasperA", ka), ("both", both)):
        W = [x for n, x in zip(names, v) if CELL[key[n][0]] == "W"]; U = [x for n, x in zip(names, v) if CELL[key[n][0]] == "U"]
        d = sorted(sum(rng.choice(W) for _ in W) / 60 - sum(rng.choice(U) for _ in U) / 60 for _ in range(10000))
        print(f"  {lab} W-U {(sum(W) - sum(U)) / 60:+.3f} [{d[250]:+.3f},{d[9749]:+.3f}]")
