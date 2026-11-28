# Deviations — disallow-canary

PREREG.md frozen at commit 54bfd401a, pushed 2026-11-28T07:01:27Z (08:01:27 CET), header by `oracle.py prereg`.

## v1.0 (before any experiment, on record)

- Agents differ from the goal's scout text: the goal named gpt-oss-20b and Qwen3-8B behind a skill gate. Both need
  Esben's laptop queue (above 2B parameters). At phase 0 the worker was "mode off" with three jobs ahead (`labq list`,
  07:53 CET), so any laptop run job would end as PENDING LAPTOP. The room chose Gemma 4 open weights
  (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) on the Gemini API before the freeze (Bruno #32365, Felix #32382, Tove
  #32372). The question says "open-weight agents" and does not name a model, so the question text is unchanged.
- The scout's v1 question and prediction were replaced by v2 after Tove's veto (#32372); v1 stays on record in the goal.

## v1.1 (facts known only after the push; filled in before the first counted run)

- Harness commit sha: TBD (Bruno)
- `mcp-server-fetch` version, scrapy, wget, trafilatura, langchain-community versions: TBD (Bruno)
- System prompt / task file sha256: TBD (Bruno)
