"""Attention on position 0 for every block and head, queries 8..127, on the 1,000 held-out sequences
(the score set), for runs measured before measure.py saved A0_all_blocks (Vera #17321).
uv run python dump_a0.py RUN:CKPT ...  -> runs/<RUN>/A0_all_blocks_c<CKPT>.npz
"""
import json, sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from gen import GenCfg, World
from measure import N_TEST, TEST_SEED, SINK_Q_MIN
from model import ModelCfg, TinyLM
HERE = Path(__file__).parent
for spec in sys.argv[1:]:
    run, ck = spec.split(":")
    log = json.loads((HERE / "runs" / run / "log.json").read_text())
    g = GenCfg(**log["gen"]); m = TinyLM(ModelCfg(**log["model"]))
    m.load_weights(str(HERE / "runs" / run / f"ckpt_{ck}.safetensors"))
    toks = World(g).sample(N_TEST[g.organism], TEST_SEED)[0]
    out = []
    for i in range(0, len(toks), 128):
        _, _, info = m.trace(mx.array(toks[i:i + 128]))
        out.append(np.stack([np.array(d["attn"][:, :, SINK_Q_MIN:, 0]) for d in info], 1).astype(np.float16))
    A = np.concatenate(out)
    np.savez_compressed(HERE / "runs" / run / f"A0_all_blocks_c{ck}.npz", A0_all_blocks=A)
    print(spec, A.shape, A.astype(np.float32).mean((0, 2, 3)).round(3))
