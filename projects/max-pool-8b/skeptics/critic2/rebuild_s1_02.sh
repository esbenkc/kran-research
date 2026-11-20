python3 - <<'PY'
import numpy as np
from pathlib import Path
for p in ['kran-research/projects/max-pool-8b/data/states/main__seed0__step150/scores.npz','kran-research/projects/max-pool-8b/data/states/clean/scores.npz']:
    z=np.load(p, allow_pickle=True)
    print(p)
    print(z.files)
    for k in z.files[:20]:
        a=z[k]
        print(' ',k, a.shape, a.dtype, a[:3] if a.ndim and a.size and a.dtype.kind in 'ifbu' else '')
PY
