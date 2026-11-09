"""Byte-identity check: our parent_hidden / nolink sites equal the parent's websites.py output (git show of the
parent branch, exec'd in a temp module). Run: .venv/bin/python test_parity.py"""
import subprocess, sys, types, filecmp, shutil
from pathlib import Path
import websites
src = subprocess.run(["git", "show", "origin/swarm/defense-1108-invisible-link:projects/invisible-link/websites.py"],
                     capture_output=True, text=True, check=True).stdout
m = types.ModuleType("parent_websites"); sys.modules["parent_websites"] = m
tmp = Path("site_parity"); m.__dict__["__file__"] = str(tmp / "x.py"); exec(src, m.__dict__)
m.SITE_DIR = tmp / "parent"
websites.SITE_DIR = tmp / "ours"
n = 0
for tpl in websites.TEMPLATE_ORDER:
    for parm, (meth, anc) in {"hidden_fit": ("parent_hidden", "fit"), "hidden_nonfit": ("parent_hidden", "nonfit"),
                              "visible_fit": ("visible", "fit"), "nolink": ("nolink", "fit")}.items():
        for stuck in (True, False):
            tok = f"{tpl}-{parm}-{stuck}"
            m.build_site(tpl, parm, stuck, token=tok, hid="abcdef012345")
            websites.build_site(tpl, meth, anc, stuck=stuck, token=tok, hid="abcdef012345")
            for f in sorted((m.SITE_DIR / tok).rglob("*.html")):
                g = websites.SITE_DIR / tok / f.relative_to(m.SITE_DIR / tok)
                assert f.read_bytes() == g.read_bytes(), (f, g); n += 1
shutil.rmtree(tmp)
print(f"PARITY OK: {n} files byte-identical to the parent")
