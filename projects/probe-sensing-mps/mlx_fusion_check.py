"""Light check (no real weights): does mx.compile fuse kernels in the mlx-lm Qwen3 forward at all?

Builds a tiny random Qwen3 (mlx-lm's own model code, 2 blocks, hidden 256), exports the graph of the eager and the
compiled forward to DOT, and counts primitive nodes and fused 'Compiled' nodes. Also compares the outputs bit for bit.

    .venv/bin/python projects/probe-sensing-mps/mlx_fusion_check.py
"""
import json
import re
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen3

HERE = Path(__file__).parent


def main() -> None:
    mx.random.seed(0)
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=256, num_hidden_layers=2, intermediate_size=768,
                           num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=1024,
                           max_position_embeddings=512, rope_theta=1e6, head_dim=64, tie_word_embeddings=True)
    model = qwen3.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    x = mx.array(np.random.default_rng(0).integers(0, 1024, (1, 71)))

    def fwd(x):
        return model(x)[0, -1]

    res = {}
    for name, fn in (("eager", fwd), ("compiled", mx.compile(fwd))):
        out = fn(x)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.dot"
            mx.export_to_dot(str(p), out)
            dot = p.read_text()
        labels = re.findall(r'label\s*=\s*"([^"]+)"', dot)
        prims = [l for l in labels if not l.startswith("{")]
        mx.eval(out)
        from collections import Counter
        res[name] = {"graph_nodes": len(prims), "compiled_nodes": sum("Compiled" in l for l in prims),
                     "node_counts": dict(Counter(re.sub(r"\s*\d+$", "", l) for l in prims)),
                     "bits": np.array(out.view(mx.uint16))}
    same = bool((res["eager"]["bits"] == res["compiled"]["bits"]).all())
    out = {k: {kk: vv for kk, vv in v.items() if kk != "bits"} for k, v in res.items()}
    out["compiled_equals_eager_bitwise"] = same
    (HERE / "data" / "mlx_fusion_check.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
