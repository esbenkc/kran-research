# /// script
# requires-python = ">=3.11"
# dependencies = ["markdownify", "html2text", "readabilipy", "trafilatura", "beautifulsoup4", "playwright", "mcp-server-fetch"]
# ///
"""Exposure census on Bruno's REAL test sites (kran-research projects/invisible-link/websites.py).

Builds every template x link arm (stuck) into a temp dir (never into the project's site/),
runs each page through the same converters as census.py, and records whether the run's
opaque hidden href (hid) and the anchor text appear. Also a <p>-wrapped toy check for trafilatura.
Run: uv run census_real.py  -> census_real.json, census_real.md
"""
import json, sys, tempfile, importlib.util, hashlib
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import census as C

PROJ = HERE.parent  # projects/invisible-link
spec = importlib.util.spec_from_file_location("websites", PROJ / "websites.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
tmp = Path(tempfile.mkdtemp()); W.SITE_DIR = tmp

def main():
    src_sha = hashlib.sha256((PROJ / "websites.py").read_bytes()).hexdigest()[:12]
    pages, meta = {}, {}
    for t in W.TEMPLATES:
        for arm in ("hidden_fit", "hidden_nonfit", "visible_fit", "sronly_fit"):
            rec = W.build_site(t, arm, stuck=True, token=f"{t}-{arm}", hid="0123456789ab")
            route = W.TEMPLATES[t]["route"][0]
            for pg in ("index.html", route):
                h = (tmp / rec["token"] / pg).read_text()
                anchor = W.TEMPLATES[t]["nonfit_anchor" if arm == "hidden_nonfit" else "fit_anchor"]
                k = f"{t}|{arm}|{pg}"
                pages[k] = h; meta[k] = (rec["hid"], anchor)
    rend = C.rendered({k.replace("|", "__").replace(".html", ""): v for k, v in pages.items()})
    rows = []
    for k, h in pages.items():
        hid, anchor = meta[k]
        outs = {n: f(h, "") for n, f in C.CONVERTERS.items()}
        outs.update(rend[k.replace("|", "__").replace(".html", "")])
        for v, o in outs.items():
            t, arm, pg = k.split("|")
            rows.append({"template": t, "arm": arm, "page": pg, "view": v,
                         "href_in_output": hid in o, "anchor_in_output": anchor in o})
    # trafilatura caveat (Kasper #16869): link wrapped in <p> inside <main>
    toy = C.page("visible", "main").replace('<div class="decoy">', "<p>").replace("</a></div>", "</a></p>")
    tp = C.trafi(toy, "")
    views = list(dict.fromkeys(r["view"] for r in rows))
    md = [f"websites.py sha256 {src_sha}; stuck sites; cell = pages (of 10: 5 templates x index + route page) with href H / anchor A present.\n",
          "| arm | " + " | ".join(views) + " |", "|---" * (len(views) + 1) + "|"]
    for arm in ("hidden_fit", "hidden_nonfit", "visible_fit", "sronly_fit"):
        cells = []
        for v in views:
            rr = [r for r in rows if r["arm"] == arm and r["view"] == v]
            cells.append(f"H{sum(r['href_in_output'] for r in rr)}/A{sum(r['anchor_in_output'] for r in rr)} of {len(rr)}")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    md.append(f"\ntrafilatura, visible link wrapped in <p> in <main> (toy page): href present = {C.TOKEN in tp}, anchor present = {C.ANCHOR in tp}")
    (HERE / "census_real.json").write_text(json.dumps({"websites_sha": src_sha, "rows": rows, "trafilatura_p_wrapped": tp}, indent=1))
    (HERE / "census_real.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
