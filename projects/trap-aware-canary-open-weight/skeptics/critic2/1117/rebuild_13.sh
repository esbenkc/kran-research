# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
from pathlib import Path
for i,line in enumerate(Path('blog.kran.ai/_notes/trap-aware-canary-open-weight.md').read_text().splitlines(),1):
    if line.startswith('figure1_alt:'):
        print(i, line)
PY
