"""Vera: harness check from raw files only (no build code imported).
Run: uv run --with markdownify==1.2.3 python vera_harness.py <project_dir> [phase]
Checks:
 1. per run: exposed = hid string in any observation the model got; follow = GET <hidden_path> status 200 in access.log
    (own parser), compared with the run file fields; leaks = GETs of other runs' /r/ paths.
 2. site bytes: where does the 2025 answer string appear (stuck: only r/ page; free: route page too);
    footer bytes per arm on index.html.
 3. md view of the footer: hidden_fit vs visible_fit (same template) after replacing the hid.
"""
import json, re, sys, os, glob, collections

proj = sys.argv[1]
phase = sys.argv[2] if len(sys.argv) > 2 else None
man = [json.loads(l) for l in open(os.path.join(proj, "raw/manifest.jsonl"))]
if phase:
    man = [m for m in man if m.get("phase") == phase]

log = collections.defaultdict(list)  # path -> [(method, status)]
for line in open(os.path.join(proj, "logs/access.log")):
    f = line.rstrip("\n").split("\t")
    if len(f) < 5:
        continue
    log[f[3].split("?")[0]].append((f[2], f[4]))
hidden_paths = {m["hidden_path"] for m in man}

bad = 0
rows = []
for m in man:
    follow = any(meth == "GET" and st == "200" for meth, st in log.get(m["hidden_path"], []))
    if m.get("raw_file") is None:  # crawler: log only
        rows.append((m["cell"], m["model"], None, follow)); continue
    rf = os.path.join(proj, m["raw_file"])
    if not os.path.exists(rf):
        print("MISSING", rf); bad += 1; continue
    d = json.load(open(rf))
    obs = [t["text"] for t in d["transcript"] if t["role"] != "assistant"]
    exposed = any(m["hid"] in o for o in obs)
    pass
    if exposed != d["exposed"] or follow != d["harness_followed"]:
        bad += 1
        print("MISMATCH", m["raw_file"], exposed, d["exposed"], follow, d["harness_followed"])
    rows.append((m["cell"], m["model"], exposed, follow))
# leaks: /r/ GETs not matching any run's own path
r_gets = [p for p in log if "/r/" in p]
unknown = [p for p in r_gets if p not in hidden_paths]
print(f"runs checked {len(rows)}, mismatches {bad}, /r/ paths in log {len(r_gets)}, not a known hidden_path {len(unknown)}")
agg = collections.defaultdict(lambda: [0, 0, 0])
for cell, model, e, f in rows:
    a = agg[(model, cell)]; a[0] += 1; a[1] += bool(e); a[2] += f
for k in sorted(agg):
    print(k, "n=%d exposed=%d followed=%d" % tuple(agg[k]))

# 2. site bytes
site = os.path.join(proj, "site")
print("\n-- answer string placement --")
for m in man[:200]:
    tdir = os.path.join(site, m["token"])
    if not os.path.isdir(tdir):
        continue
    hits = [os.path.relpath(p, tdir) for p in glob.glob(tdir + "/**/*.html", recursive=True)
            if m["answer"] in open(p).read()]
    key = (m["stuck"], m["arm"], tuple(sorted(h if not h.startswith("r/") else "r/*" for h in hits)))
    agg2 = globals().setdefault("agg2", collections.Counter()); agg2[key] += 1
for k, v in sorted(globals().get("agg2", {}).items()):
    print(k, v)

print("\n-- footer per arm (index.html, hid masked) --")
seen = {}
for m in man:
    p = os.path.join(site, m["token"], "index.html")
    if not os.path.exists(p):
        continue
    html = open(p).read()
    foot = re.search(r"<footer>.*?</footer>", html, re.S)
    foot = foot.group(0).replace(m["hid"], "HID") if foot else "NO FOOTER"
    seen.setdefault((m["template"], m["arm"]), foot)
for k in sorted(seen):
    if k[0] == "pricing":
        print(k, seen[k])

try:
    from markdownify import markdownify as md
    print("\n-- md footer: hidden_fit vs visible_fit, per template --")
    for t in sorted({k[0] for k in seen}):
        a, b = seen.get((t, "hidden_fit")), seen.get((t, "visible_fit"))
        if a and b:
            print(t, "md identical:", md(a) == md(b), "| raw identical:", a == b)
except ImportError:
    print("markdownify not installed")
