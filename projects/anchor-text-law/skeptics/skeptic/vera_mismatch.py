# Vera: list runs where my access-log follow differs from the harness's harness_followed flag.
import vera_check as V
a, _ = V.model_dirs()
for d in a + _:
    runs = V.load_runs(d); acc = V.load_access(d)
    for c in runs:
        for i, r in sorted(runs[c].items()):
            hid = r['site']['hid']
            hits = [(p, l, m) for p, l, m in acc if hid in p]
            if bool(hits) != bool(r['harness_followed']):
                print(d.split('/')[-1], c, i, 'mine', bool(hits), 'harness', r['harness_followed'], hits[:3],
                      [x['path'] for x in r['actions'] if hid in (x.get('path') or x.get('url') or '')])
