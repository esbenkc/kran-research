# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/raw/warning-cued/030.json').read_text())
for name in ['actions','replies','transcript']:
    print('\n',name, type(r.get(name)).__name__)
    obj=r.get(name)
    if isinstance(obj,list):
        for i,x in enumerate(obj[:5]):
            s=json.dumps(x, ensure_ascii=False) if not isinstance(x,str) else x
            if any(term in s.lower() for term in ['666221','trap','display:none','footer']):
                print(i, s[:2000].replace('\n',' '))
    elif isinstance(obj,str):
        idx=min([i for i in [obj.lower().find('666221'),obj.lower().find('trap'),obj.lower().find('display:none'),obj.lower().find('footer')] if i>=0] or [-1])
        print(obj[max(0,idx-500):idx+1000].replace('\n',' '))
PY
