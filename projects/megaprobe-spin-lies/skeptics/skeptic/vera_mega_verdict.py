"""PREREG s6 quantities on MY OWN arms (vera_mega c' vs vera_mega a, own folds seed 77), same paired prompt bootstrap.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_mega_verdict.py vera_mega_fs77_d3_sh0_own.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import TRAINED, auroc, classes, load_labels  # noqa: E402
from vera_io import D20  # noqa: E402
from vera_verdict import verdict  # noqa: E402

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / sys.argv[1]))
LAB = {m: load_labels(m, D20)[:1000] for m in d}
S = {m: {"c": np.array(d[m]["sc"]), "a": np.array(d[m]["sa"])} for m in d}
if "base" in d:
    c = classes(LAB["base"])
    print("base all lies: c' %.4f vs a %.4f" % (auroc(S["base"]["c"][c["lie"]], S["base"]["c"][c["truthful"]]),
                                               auroc(S["base"]["a"][c["lie"]], S["base"]["a"][c["truthful"]])))
for neg in ["truthful_fs", "truthful"]:
    v = verdict({m: S[m] for m in TRAINED}, {m: LAB[m] for m in TRAINED}, neg=neg)
    print(f"{sys.argv[1]} neg={neg}: m(c') {v['m']:.4f} [{v['m_lo']:.4f},{v['m_hi']:.4f}] g(c'-a) {v['g']:+.4f} [{v['g_lo']:+.4f},{v['g_hi']:+.4f}] "
          f"raw-rule {v['status']} | " + " ".join(f"{k} {x:.3f}" for k, x in v["per_model"].items()))
lies = [auroc(S[m]["c"][classes(LAB[m])["lie"]], S[m]["c"][classes(LAB[m])["truthful"]]) for m in TRAINED]
liesa = [auroc(S[m]["a"][classes(LAB[m])["lie"]], S[m]["a"][classes(LAB[m])["truthful"]]) for m in TRAINED]
print("trained all lies mean: c' %.4f a %.4f" % (np.mean(lies), np.mean(liesa)))
