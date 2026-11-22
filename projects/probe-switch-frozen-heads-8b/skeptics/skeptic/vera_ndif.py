#!/usr/bin/env python3
"""Vera: NDIF (NNsight, CUDA bf16) vs local (CPU fp32 eager) untrained attention mass from the answer position to
positions 70-78, per head of blocks 28-35. Usage: python3 vera_ndif.py PROJECT_DIR"""
import json, sys
P = sys.argv[1].rstrip('/') + '/'
n = json.load(open(P + 'crosscheck_ndif/attn_base.json'))['mass_70_78']
r = json.load(open(P + 'research_checks/route.json'))['heads']
common = [h for h in n if h in r]
d = sorted(((abs(n[h] - r[h]['to_70_78']), h, n[h], r[h]['to_70_78']) for h in common), reverse=True)
print(f"heads compared {len(common)}; max |NDIF - local| {d[0][0]:.4f} at {d[0][1]} ({d[0][2]:.4f} vs {d[0][3]:.4f})")
ge = lambda m: sorted(h for h, v in m.items() if v >= 0.30 and int(h.split('.')[0]) >= 29)
gn, gr = ge(n), ge({h: r[h]['to_70_78'] for h in r})
print(f"heads >= 0.30 in blocks 29-35: NDIF {len(gn)}, local {len(gr)}, same set {gn == gr}")
