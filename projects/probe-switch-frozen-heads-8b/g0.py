"""G0 (PREREG J0 'G0 trace check'): does the read-only probe hook change the cached block-FIRST input (whole compiled
graph)? Per split: share of prompts where the answer position (79) differs in any bit, share where any position
differs, mean number of changed coordinates at the answer position, and the mean relative size of the change there.
Also: share of prompts whose untrained answer-token logits differ in any bit between the two compiled builds.

    $PY g0.py      -> DATA/g0.json (prints one summary line)
"""
import json

import numpy as np

from cfgm import CACHE, DATA, FIRST, PROMPTS


def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    z0 = np.load(CACHE / "none_A_tailin.npy", mmap_mode="r")
    z1 = np.load(CACHE / "probe_A_tailin.npy", mmap_mode="r")
    l0 = np.load(CACHE / "none_A_logits.npy", mmap_mode="r")
    l1 = np.load(CACHE / "probe_A_logits.npy", mmap_mode="r")
    pid = json.load(open(CACHE / "meta_A.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(PROMPTS)}
    out = {"site": f"in{FIRST}", "by_split": {}}
    for s in ("train", "dev", "test"):
        ids = [i for i, p in enumerate(pid) if split[p] == s]
        a0, a1 = np.asarray(z0[ids]), np.asarray(z1[ids])
        ans0, ans1 = f32(a0[:, 79]), f32(a1[:, 79])
        out["by_split"][s] = {
            "n": len(ids),
            "share_answer_pos_differs": float(np.mean((a0[:, 79] != a1[:, 79]).any(-1))),
            "share_any_pos_differs": float(np.mean((a0 != a1).reshape(len(ids), -1).any(-1))),
            "mean_changed_dims_answer": float(np.mean((a0[:, 79] != a1[:, 79]).sum(-1))),
            "mean_rel_change_answer": float(np.mean(np.linalg.norm(ans1 - ans0, axis=-1) / np.linalg.norm(ans0, axis=-1))),
            "share_logits_differ": float(np.mean((np.asarray(l0[ids]) != np.asarray(l1[ids])).any(-1))),
        }
    out["d_ans"] = out["by_split"]["test"]["share_answer_pos_differs"]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "g0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
