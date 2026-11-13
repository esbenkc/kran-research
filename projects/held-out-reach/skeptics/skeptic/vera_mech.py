"""Vera: decompose each route combo into  tried-the-route -> route showed the address -> followed.
Route actions: lcpw extract_hyperlinks/get_elements; pwmcp browser_evaluate; bufull evaluate (+extract logged).
'showed' = the action's observation held the run's hidden id (hid_in_obs, cross-checked against the transcript text).
Also: for every follow, which action first exposed the hid before the follow."""
import collections, glob, json, os

PROJ = os.path.expanduser("~/work/defense-1113-held-out-reach-law/kran-research/projects/held-out-reach")
ROUTE = {"lcpw": {"extract_hyperlinks", "get_elements"}, "pwmcp": {"browser_evaluate"}, "bufull": {"evaluate"},
         "mcpfetch": {"fetch"}}

tab = collections.defaultdict(lambda: collections.Counter())
first_expose = collections.defaultdict(collections.Counter)
code_kinds = collections.defaultdict(collections.Counter)
for p in sorted(glob.glob(f"{PROJ}/raw/counted/*/*__display_none__fit/*.json")):
    d = json.load(open(p))
    m, cell = p.split("/")[-3], p.split("/")[-2]
    tool = cell.split("__")[0]
    if tool not in ROUTE:
        continue
    hid = d["site"]["hid"]
    acts = d["actions"]
    fidx = next((i for i, a in enumerate(acts) if hid in (a.get("url") or "")), None)
    if tool == "mcpfetch":
        tried = any(a["tool"] == "fetch" and (a.get("args") or {}).get("raw") in (True, "true") for a in acts)
    else:
        tried = any(a["tool"] in ROUTE[tool] for a in acts)
    obs = [t["text"] for t in d["transcript"] if t["role"] == "observation"]
    showed_flag = any(a.get("hid_in_obs") for a in acts)
    showed_txt = any(hid in o for o in obs)
    t = tab[(tool, m)]
    t["n"] += 1
    t["tried"] += tried
    t["showed"] += showed_txt
    t["showed_flag_mismatch"] += showed_flag != showed_txt
    t["followed"] += fidx is not None
    t["followed_without_show"] += (fidx is not None) and not showed_txt
    t["showed_not_followed"] += showed_txt and fidx is None
    if fidx is not None:
        ex = next((a["tool"] for a in acts[:fidx] if a.get("hid_in_obs")), "none-before-follow")
        first_expose[tool][ex] += 1
    for a in acts:
        if a["tool"] in ("browser_evaluate", "evaluate"):
            code = json.dumps(a.get("args"))
            kind = ("outerHTML/innerHTML" if ("outerHTML" in code or "innerHTML" in code) else
                    "querySelectorAll(a)/href" if ("href" in code or "querySelectorAll" in code or ".links" in code) else
                    "innerText/textContent" if ("innerText" in code or "textContent" in code) else "other")
            code_kinds[tool][(kind, bool(a.get("hid_in_obs")))] += 1

for k in sorted(tab):
    t = tab[k]
    print(f"{k[0]:8s} {k[1]:24s} n={t['n']:3d} tried={t['tried']:3d} showed={t['showed']:3d} followed={t['followed']:3d} "
          f"| followed_w/o_show={t['followed_without_show']} showed_not_followed={t['showed_not_followed']} flag_mismatch={t['showed_flag_mismatch']}")
print("\nfirst action that exposed the hid before each follow:")
for tool, c in first_expose.items():
    print(" ", tool, dict(c))
print("\nevaluate code kinds (kind, showed hid): count")
for tool, c in code_kinds.items():
    print(" ", tool, dict(c))
