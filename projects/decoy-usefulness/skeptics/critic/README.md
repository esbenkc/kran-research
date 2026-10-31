# Critic (Kasper) rebuild receipts

Kasper's own code (written read-only in /tmp/kasper/k1031, copied here unchanged by Bruno at his request, #10292).
Run from projects/decoy-usefulness:

```sh
python3 skeptics/critic/rebuild.py raw > skeptics/critic/rebuild.out   # own scorer from raw JSONL; primary and credential-only
python3 skeptics/critic/edge.py > skeptics/critic/edge.out             # exact bootstrap tail at the 0.15 band edge
python3 skeptics/critic/onefewer.py > skeptics/critic/onefewer.out     # one run more/less in each off-diagonal cell
```
