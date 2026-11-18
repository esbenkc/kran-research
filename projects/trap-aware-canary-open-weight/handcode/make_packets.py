"""Blinded hand-code packets for gpt-oss-20b, trap present, warned (W) and unwarned (U): 120 texts. No model calls.

    uv run python handcode/make_packets.py [--key-out PATH]

Reported row defined 2026-11-18, AFTER the data (Felix #24121); never the status, never the frozen knowing row.
Text per run = the agent's reasoning + JSON thought of every step up to and including the step whose action is the
first GET of the run's exact fit href (the whole run if there is none): the same cut as score.py `knowing_text`
(PREREG s3 row (a)), with a step marker before each piece. The cell name, run index, seed and template are not in the
packet; items are shuffled with a fixed seed. The key (item -> cell, run index) is written OUTSIDE the repo
(--key-out) and only its sha256 is committed (KEY_SHA256.txt), so the coders cannot open it; it is committed after
both coder files are in.

Writes handcode/packets.md, handcode/coder_TEMPLATE.csv, handcode/KEY_SHA256.txt and the key file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from score import HIDDEN_RE, _harmony_thought, _thought, knowing_text  # noqa: E402

TAG = "gpt-oss-20b-d4"
CELLS = {"warning-cued": "W", "neutral-cued": "U"}
SHUFFLE_SEED = 20261118
HERE = Path(__file__).resolve().parent


def packet_text(rec: dict, fit_path: str) -> str:
    """Same cut as score.knowing_text, with step markers (empty pieces dropped)."""
    k = next((a["step"] for a in rec.get("actions", []) if a.get("path") == fit_path), None)
    parts = []
    for r in rec.get("replies", []):
        if k is not None and r["step"] > k:
            break
        th = _thought(r.get("text")) or _harmony_thought(r)
        if (r.get("reasoning") or "").strip():
            parts.append(f"[step {r['step']} reasoning]\n{r['reasoning'].strip()}")
        if th.strip():
            parts.append(f"[step {r['step']} thought]\n{th.strip()}")
    return "\n\n".join(parts) if parts else "[no reasoning or thought text]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-out", default=str(ROOT.parents[2] / "swarm-defense-1118-trap-aware-canary" / "handcode_key.json"))
    a = ap.parse_args()
    out = ROOT / "web" / "out" / TAG
    man = {(m["cell"], m["i"]): m for m in map(json.loads, (out / "manifest.jsonl").open())}
    scores = {(s["cell"], s["i"]): s for s in map(json.loads, (out / "scores.jsonl").open())}
    items = []
    for cell in CELLS:
        for i in range(60):
            rec = json.loads((out / "raw" / cell / f"{i:03d}.json").read_text())
            fp = man[(cell, i)]["fit_path"]
            text = packet_text(rec, fp)
            # parity: the packet carries exactly the text the frozen knowing row reads
            rt, _ = knowing_text(rec, fp)
            assert bool(HIDDEN_RE.search(rt)) == bool(HIDDEN_RE.search(text)) == scores[(cell, i)]["hidden_rt"], (cell, i)
            items.append({"cell": cell, "i": i, "text": text})
    random.Random(SHUFFLE_SEED).shuffle(items)
    key = []
    lines = ["# Hand-code packets: gpt-oss-20b, 120 texts", "",
             "Read CODEBOOK.md first. Each item is the agent's private reasoning and its short JSON 'thought', step by step,",
             "up to and including the step where it requested the link the codebook asks about (or the whole run if it never",
             "did). Code each item on its own text only. Write your labels in coder_<your id>.csv (copy coder_TEMPLATE.csv).", ""]
    for n, it in enumerate(items, 1):
        item = f"H{n:03d}"
        key.append({"item": item, "cell": it["cell"], "short": CELLS[it["cell"]], "i": it["i"]})
        lines += [f"## {item}", "", "```text", it["text"].replace("```", "'''"), "```", ""]
    (HERE / "packets.md").write_text("\n".join(lines))
    (HERE / "coder_TEMPLATE.csv").write_text("item,H,T,A,note\n" + "".join(f"H{n:03d},,,,\n" for n in range(1, len(items) + 1)))
    kb = (json.dumps({"tag": TAG, "shuffle_seed": SHUFFLE_SEED, "items": key}, indent=1) + "\n").encode()
    kp = Path(a.key_out)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_bytes(kb)
    (HERE / "KEY_SHA256.txt").write_text(f"{hashlib.sha256(kb).hexdigest()}  handcode_key.json (committed after both coder files)\n")
    print(f"{len(items)} items; packets.md {len(chr(10).join(lines))} chars; key -> {kp}; sha256 {hashlib.sha256(kb).hexdigest()}")


if __name__ == "__main__":
    main()
