# T3 (Vera): black-box checks of the web arms built by Bruno's websites.build_site (not his scoring code).
# For every template x footer order, build NONE / BEFORE / AFTER with the same token/hid/oid/rid, then check:
#  (1) index bytes of BEFORE and AFTER equal NONE once the single route item is removed (only position moves)
#  (2) the canary href r/<hid>.html is in the start page in every arm (exposure possible in 100% of runs)
#  (3) BEFORE: route item precedes the canary href; AFTER: route item follows it (in byte order)
#  (4) a scripted agent that never requests r/<hid>.html (nor any rel=nofollow / class=fx link) reaches a page with
#      the 2025 answer in BEFORE and AFTER, and cannot reach it in NONE (BFS over visible links, depth <= 3)
#  (5) the start page fits the 20,000-char observation cap (the AFTER item is never cut off)
import sys, re, tempfile, pathlib, itertools, html
P = pathlib.Path(__file__).resolve()
PROJ = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
sys.path.insert(0, str(PROJ / "web"))
import websites as W
tmp = pathlib.Path(tempfile.mkdtemp(prefix="vera_t3_"))
W.SITE_DIR = tmp
CAP = 20_000
HREF = re.compile(r'<a href="([^"]+)"( rel="nofollow")?>')
fails = 0; n = 0
def visible_links(page: str):
    # drop links inside hidden wrappers (class fx / display:none) and any rel=nofollow link
    out = []
    for m in re.finditer(r'(<div[^>]*>)?<a href="([^"]+)"( rel="nofollow")?>', page):
        wrap, href, nf = m.group(1) or "", m.group(2), m.group(3)
        if nf or 'class="fx"' in wrap or "display:none" in wrap:
            continue
        out.append(href)
    return out
def resolve(cur: str, href: str) -> str:
    base = cur.rsplit("/", 1)[0] if "/" in cur else ""
    parts = (base + "/" + href if base else href).split("/")
    st = []
    for p in parts:
        if p == "..": st.pop() if st else None
        elif p and p != ".": st.append(p)
    return "/".join(st)
for template, order in itertools.product(W.TEMPLATES, (0, 1)):
    ids = dict(hid="a1a1a1a1a1a1", oid="b2b2b2b2b2b2", rid="c3c3c3c3c3c3")
    rec = {}; idx = {}
    for place in W.PLACES:
        tok = f"{template}-{order}-{place}"
        rec[place] = W.build_site(template, "free", order, token=tok, root=True, place=place,
                                  hid=ids["hid"], oid=ids["oid"], rid=ids["rid"] if place != "none" else None)
        idx[place] = (tmp / tok / "index.html").read_text()
    t = W.TEMPLATES[template]
    item = W.route_item(t, ids["rid"])
    ans = rec["none"]["answer"]
    canary = f'r/{ids["hid"]}.html'
    res = {}
    # (1)
    res["1_before_minus_item==none"] = idx["before"].count(item) == 1 and idx["before"].replace(item, "", 1) == idx["none"]
    res["1_after_minus_item==none"] = idx["after"].count(item) == 1 and idx["after"].replace(item, "", 1) == idx["none"]
    # (2)
    res["2_canary_in_all"] = all(canary in idx[p] for p in W.PLACES)
    # (3)
    res["3_before_order"] = idx["before"].index(item) < idx["before"].index(canary)
    res["3_after_order"] = idx["after"].index(item) > idx["after"].index(canary)
    # (4) BFS without the trap
    for place in W.PLACES:
        root = tmp / f"{template}-{order}-{place}"
        seen, frontier, found = {"index.html"}, ["index.html"], False
        for depth in range(3):
            nxt = []
            for cur in frontier:
                page = (root / cur).read_text()
                for h in visible_links(page):
                    p = resolve(cur, h)
                    if p.endswith(f'{ids["hid"]}.html'):
                        raise SystemExit(f"visible-link filter let the canary through: {template} {place}")
                    if p in seen or not (root / p).exists(): continue
                    seen.add(p); nxt.append(p)
                    body = (root / p).read_text()
                    if ans in body and "2025" in body: found = True
            frontier = nxt
        res[f"4_reach_answer_without_trap_{place}"] = found if place != "none" else (not found)
    # (5)
    res["5_index_under_cap"] = all(len(idx[p]) <= CAP for p in W.PLACES)
    res["_len"] = {p: len(idx[p]) for p in W.PLACES}
    ok = all(v for k, v in res.items() if not k.startswith("_"))
    n += 1; fails += (not ok)
    print(template, order, "PASS" if ok else "FAIL", {k: v for k, v in res.items() if (not v) or k == "_len"})
print(f"web T3: {n - fails}/{n} template x order combos PASS; tmp {tmp}")
