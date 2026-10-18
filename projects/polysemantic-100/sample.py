"""Pre-registered random sample of neurons. Run once, before any activation is read.

    uv run python projects/polysemantic-100/sample.py   -> data/neurons.json
"""
import json

import numpy as np

from common import CFG, DATA


def main() -> None:
    L, D = CFG["n_layers"], CFG["d_mlp"]
    rng = np.random.default_rng(CFG["sample"]["seed"])
    flat = rng.choice(L * D, size=CFG["sample"]["n"], replace=False)
    neurons = [{"id": f"L{int(f) // D}N{int(f) % D}", "layer": int(f) // D, "neuron": int(f) % D}
               for f in flat]
    neurons.sort(key=lambda n: (n["layer"], n["neuron"]))
    DATA.mkdir(exist_ok=True)
    (DATA / "neurons.json").write_text(json.dumps(neurons, indent=1))
    print(len(neurons), "neurons, layers", sorted({n["layer"] for n in neurons}))


if __name__ == "__main__":
    main()
