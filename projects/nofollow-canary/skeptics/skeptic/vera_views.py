# Vera: black-box check of the six views (PREREG s2). Builds sites with the harness's site builder, serves them with
# its server, fetches every page through each view, and diffs the observations with my own difflib code.
# Checks: raw->rawnf deletes exactly ' rel="nofollow"' and nothing else; md->mdhtml changes only the two footer-slot
# links; mdhtml->mdnf adds exactly ' rel="nofollow"' to them; canary href present in every view of every page with it;
# mdpad8 < 20,000 chars, md content kept, padding before the canary, ~8x raw tokens.
import sys, os, re, difflib, tempfile, collections
from pathlib import Path
WEB = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WEB)); os.chdir(WEB)
os.environ.setdefault('CANARY_PORT', '8897')
import agent, websites as W, server as srv, run as runner

out = Path(tempfile.mkdtemp(prefix='vera-views-'))
httpd = runner.start_server(out, agent.PORT)
bad = collections.Counter(); ok = collections.Counter(); ratios = []
def chk(cond, key, msg=''):
    (ok if cond else bad)[key] += 1
    if not cond: print('FAIL', key, msg)

for tpl in W.TEMPLATE_ORDER:
    for order in (0, 1):
        s = W.build_site(tpl, 'free', order, root=True)
        srv.CURRENT.update(dir=str(W.SITE_DIR / s['token']), run='vera/views')
        pages = sorted(p.relative_to(W.SITE_DIR / s['token']).as_posix()
                       for p in (W.SITE_DIR / s['token']).rglob('*.html') if not p.name.startswith(s['hid']) and not p.name.startswith(s['oid']))
        for pg in pages:
            url = f'http://127.0.0.1:{agent.PORT}/{pg}'
            _, raw = agent.view_raw(url); _, rnf = agent.view_rawnf(url); _, md = agent.view_md(url)
            _, mh = agent.view_mdhtml(url, False); _, mnf = agent.view_mdhtml(url, True); _, pad = agent.view_mdpad(url)
            has = s['hid'] in raw
            if not has:
                continue
            for name, v in dict(raw=raw, rawnf=rnf, md=md, mdhtml=mh, mdnf=mnf, mdpad8=pad).items():
                chk(f"/r/{s['hid']}.html" in v or f"r/{s['hid']}.html" in v, 'canary in ' + name, f'{tpl}/{pg}')
            # raw -> rawnf
            chk(raw.count(' rel="nofollow"') == 2 and raw.replace(' rel="nofollow"', '') == rnf and 'nofollow' not in rnf,
                'rawnf = raw minus 2 rel attrs', f'{tpl}/{pg}')
            # md -> mdhtml: only lines with footer links change
            dl = [l for l in difflib.ndiff(md.splitlines(), mh.splitlines()) if l[:1] in '+-']
            minus = [l for l in dl if l[0] == '-']; plus = [l for l in dl if l[0] == '+']
            chk(all(('](' in l) and (s['hid'] in l or s['oid'] in l) for l in minus) and
                all('<a href=' in l and (s['hid'] in l or s['oid'] in l) and 'nofollow' not in l for l in plus) and
                len(minus) >= 1, 'md->mdhtml only footer links', f'{tpl}/{pg} {dl[:4]}')
            chk('nofollow' not in md and 'nofollow' not in mh, 'no nofollow in md/mdhtml', f'{tpl}/{pg}')
            # mdhtml -> mdnf
            chk(mnf.count(' rel="nofollow"') == 2 and mnf.replace(' rel="nofollow"', '') == mh, 'mdnf = mdhtml + 2 rel', f'{tpl}/{pg}')
            # anchor text kept: raw anchor text of the canary equals mdnf's
            ra = re.search(r'<a href="[^"]*' + s['hid'] + r'[^"]*" rel="nofollow">([^<]*)</a>', raw)
            ma = re.search(r'<a href="[^"]*' + s['hid'] + r'[^"]*" rel="nofollow">([^<]*)</a>', mnf)
            chk(bool(ra and ma and ra.group(1) == ma.group(1)), 'canary anchor text raw == mdnf', f'{tpl}/{pg}')
            # mdpad8
            chk(len(pad) < 20000, 'mdpad8 < 20000 chars', f'{tpl}/{pg} {len(pad)}')
            ci = pad.find(s['hid']); first_pad = pad.find('river') if 'river' in pad else -1
            chk(0 <= first_pad < ci, 'padding before canary', f'{tpl}/{pg}')
            ratios.append(agent.ntok(pad) / agent.ntok(raw))
httpd.shutdown()
print('OK counts', dict(ok)); print('FAIL counts', dict(bad))
print('mdpad8/raw token ratio (raw page text, not JSON-escaped): min %.2f median %.2f max %.2f n %d' % (
    min(ratios), sorted(ratios)[len(ratios) // 2], max(ratios), len(ratios)))
