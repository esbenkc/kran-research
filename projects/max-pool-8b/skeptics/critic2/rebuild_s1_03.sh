python3 - <<'PY'
import numpy as np, json
from pathlib import Path
for p in ['kran-research/projects/max-pool-8b/data/states/clean/index.npz','kran-research/projects/max-pool-8b/data/ref/split.json']:
    print(p, Path(p).exists())
if Path('kran-research/projects/max-pool-8b/data/states/clean/index.npz').exists():
    z=np.load('kran-research/projects/max-pool-8b/data/states/clean/index.npz', allow_pickle=True)
    print(z.files)
    for k in z.files:
        a=z[k]
        print(k, a.shape, a.dtype, a[:10] if getattr(a,'ndim',0) and a.size and a.dtype.kind in 'ifbuiUOS' else '')
PY
