"""Export the raw numbers a rebuild needs into data/ (git), from raw/ (local only).

data/behav/<policy>.npz   per build x test prompt: px (exact first-token P(X)), top (top-1 id); arrays 'builds', 'ids'
                          (policy = base | the run name of each trained policy); plus <policy>_sha.json (build -> the
                          200 last-row logit sha256[:16], for the bit-exact check)
data/behav/<policy>_none_gen.json  the 8-token greedy text of the plain none graph per test prompt (headline policies)
data/traces/<tag>.npz     block-26 input at position 25 ('<|im_end|>') in the base rec pass, bf16 bits as uint16
                          [582, 1024] for the train + test prompts ('ids' gives the order); every non-bit-exact build,
                          plus none and A. Overlap rebuild: fp32 = (u.astype(uint32) << 16).view(float32).

    python3 export.py
"""
import json
from pathlib import Path

import numpy as np

from fp import read_builds, tagof

HERE = Path(__file__).parent
POLICIES = {"base": "raw/screen/base"} | {r: f"raw/behav/{r}" for r in (
    "r50_s0_lr0.0001_kl1_at8192", "r50_s1_lr0.0001_kl1_at8192", "r50_s0_lr0.0001_kl1_at4096",
    "r50_s1_lr0.0001_kl1_at4096", "r50_s0_lr0.0001_kl1", "shuf_s0_lr0.0001_kl1")}


def main() -> None:
    order = read_builds(HERE / "data" / "builds_frozen.txt")
    (HERE / "data" / "behav").mkdir(parents=True, exist_ok=True)
    for name, d in POLICIES.items():
        d = HERE / d
        bs = [b for b in order if (d / f"{tagof(b)}_plain_info.json").exists()]
        if not bs:
            continue
        z = [np.load(d / f"{tagof(b)}_plain.npz") for b in bs]
        ids = z[0]["ids"]
        assert all((x["ids"] == ids).all() for x in z)
        np.savez_compressed(HERE / "data" / "behav" / f"{name}.npz", builds=np.array(bs), ids=ids,
                            px=np.stack([x["px"] for x in z]), top=np.stack([x["top"] for x in z]))
        json.dump({b: json.load(open(d / f"{tagof(b)}_plain_sha.json")) for b in bs},
                  open(HERE / "data" / "behav" / f"{name}_sha.json", "w"), separators=(",", ":"))
        g = d / "none_plain_gen.json"  # 8-token greedy text of the plain none graph (text judge, Figure 1 example)
        if g.exists():
            json.dump(json.load(open(g)), open(HERE / "data" / "behav" / f"{name}_none_gen.json", "w"))
        print(name, len(bs), "builds")
    ov = json.load(open(HERE / "data" / "overlap.json"))["builds"]
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    keep = np.array([p["split"] != "dev" for p in prompts])
    ids = np.array([p["id"] for p in prompts])[keep]
    (HERE / "data" / "traces").mkdir(parents=True, exist_ok=True)
    n = 0
    for b in order:
        if b in ("none", "ref@L14.q_norm") or ov[b]["class"] != "bitexact":
            u = np.load(HERE / "raw" / "screen" / "base" / f"{tagof(b)}_rec_tail.npy", mmap_mode="r")[:, 25, :][keep]
            np.savez_compressed(HERE / "data" / "traces" / f"{tagof(b)}.npz", u=np.ascontiguousarray(u), ids=ids)
            n += 1
    print("traces", n)


if __name__ == "__main__":
    main()
