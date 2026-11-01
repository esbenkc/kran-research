# redirect-budget: DEVIATIONS (Bruno, build)

Every departure from PREREG.md, with the time it happened (PST) and how far it can move the result.

- **D1 (06:05, before bounds.py existed).** To validate capture.py, I ran it on the base 0.6B model and on the
  monitor_s0 copy and printed the measured answer-token TV of the 16 block-27 heads (27.8: 0.659, 27.9: 0.31,
  the rest <= 0.013) and head 27.8's attention on tokens 0-2 (matches sink-value's data/attn_onoff.json, 0.80 vs
  0.807). The same numbers were already public in the sink-value post. No bound was computed then; the carrier
  rule and the bound rows were frozen in PREREG v1-v1.2 before that and do not depend on it. Effect: none.
- **D2 (06:06).** carriers06.py also scores the dev split of the cached 0.6B file (64 prompts outside the 200)
  as a cross-check row. The carrier is the argmax on the 200 prompts, as PREREG sec 1-2 says; the dev argmax is
  the same head on all 4 copies. Effect: none.
- **D3 (06:06).** J0 (cache.py --pass A) ran with its /usr/bin/time wrapper killed when the tool call returned
  (python survived and finished), so its peak RSS was not recorded. Both caches match manifest_raw.json sha256
  bit for bit. Effect: none on results; memory peak known from the source room (5.1 GB).
