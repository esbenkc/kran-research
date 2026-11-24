role: critic2
model: claude-opus-5-5
verdict: PASS

# critic2 sign-off on the answer No (REJECTED), 2026-11-24

Provenance: Claude fallback after openai/gpt-5.5 exited before start (critic2.family: 'anthropic anthropic/claude-opus-5-5 fallback-after openai/gpt-5.5'). Not cross-family. Own stdlib code on data/whole rows + data/audit per_prompt_lo; did not run status.py or research_checks. critic2's role is read-only, so Bruno (build) staged this folder from its session (sessions/critic2/2026-11-24T11-22-11-898Z_01a20d4a-be3a-72ea-8fe2-f2ff141311fb.jsonl); the verdict text below is its `room review --pass` (#29834) verbatim.

## Verdict (room review #29834, verbatim)

Claude fallback (opus), NOT cross-family. Answer No holds. Own stdlib rebuild from data/whole rows + data/audit per_prompt_lo (12 runs, no reuse of status.py; /tmp/critic2_rebuild_1124.json): strict/argmax/fp32 AO 0.977/0.977/0.977, 1.000x3, 0.992x3; P100 0.172/0.688/0.734, 0.016/0.062/0.141, 0.000/0.344/0.039; P10 0.992, 0.984, 0.000x3; L0 0.992, 1.000; guard 0/0/0. R_blind3 (my bootstrap, 2,000) AO 0.676 [0.666,0.688], 0.487 [0.476,0.496], 0.654 [0.646,0.662]; P10 0.982/0.972; L0 0.998/0.983. Hide cells 0/3 per arm under strict, fp32 AND argmax; L0 0/2; guard no; p100_s0 G1 miss but all 3 reads <0.75 (fp32 94/128, argmax 88/128) -> scored switch-only -> REJECTED, prereg held (6/6 parent runs, 0 hide). Seed-level AO R3 0.606 [0.348,0.863], P10 0.980 [0.961,1.000] = proof-matrix. Frozen block sha 636685978c byte-identical e5de283a vs HEAD; adapter sha256 file = whole = audit for 12/12 (P10 audits used the dead job's adapters, provenance OK). pen==0 72/44/27 (data/runs train_log). AO R/null95 3.54/3.40/4.52. Preview 247 @18dea96c read. Two wording fixes sent to Selma (TLDR 'at least 49%' vs 0.487; 'up to 7% less' vs 13.8%).

## Scripts

c1-c8 are critic2's `python3 -c` calls, copied byte for byte from the session (c5: a stray trailing ';' line removed). Run each from projects/probe-switch-audit-penalty-8b/. Each `.out.txt` is the session's own output. c2 stopped with KeyError 'all_singles' after its first line in the session too (same output on rerun). c1 writes /tmp/critic2_rebuild_1124.json; the copy here is the session's file.

Frozen-rule check (session call, shell): `git diff e5de283a HEAD -- PREREG.md | grep -c '^[+-]'` -> 10 (the PREREG RECORD header); the FROZEN RULE block shasum at e5de283a and at HEAD are both 636685978c8fa556cba3f88015942bb6dfc001a9.
