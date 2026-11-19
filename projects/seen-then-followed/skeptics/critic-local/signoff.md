role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: seen-then-followed

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 31 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-19T10:52:22+00:00 on Mac, mlx-lm 0.32.2, reasoning effort medium, seed 0, 42443 prompt tokens, 3903 generated, peak memory 13.92 GB, 174.8 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, results/runs_gate.jsonl, results/runs_main.jsonl, results/analysis.json, results/crawl.json, results/discarded_attempts.json, results/switch.json, results/test_harness.txt, slots.json, config.yaml and 23 more output files; 625 smaller output files left out for length.

## The claim it checked

The hidden canary catch rate is not simply the product of a tool’s reach and a model’s follow rate: Gemma 4 26B’s follow rate does not carry across fetch tools, leading to a ~30‑point failure for one held‑out pair.

## Does the data support the claim? yes

The frozen‑rule analysis (results/analysis.json) reports the status as REJECTED because |err|>0.10 in one of the held‑out cells.  The post’s narrative that the rule missed one pair by ~30 % and that the other pair’s prediction error was ~11 % matches the extraction error values 0.2963 and 0.1111, respectively, and all reported counts agree with the run‑level tallies in the same file.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| Gemma 4 26B saw the link’s address in all 81 runs but requested it in only 57 (70 %) | 57 of 81 | 57 of 81 | results/analysis.json | yes | yes |
| –30 % error (predicted 100 % vs measured 70 %) | 0.30 | 0.296296 | results/analysis.json | yes | yes |
| Gemma 4 31B with MCP fetch tool measured 80 % (65 of 81) | 65 of 81 | 65 of 81 | results/analysis.json | yes | yes |
| –11 % error (predicted 69 % vs measured 80 %) | 0.11 | 0.111111 | results/analysis.json | yes | yes |
| Predicted 69 % for that held‑out pair | 0.69 | 0.691358 | results/analysis.json | yes | yes |
| Predicted 100 % for the other held‑out pair | 1.00 | 1.000000 | results/analysis.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The primary threat is the implicit assumption that a model’s follow rate is invariant across fetch tools.  The analysis shows a 30 % discrepancy in follow for Gemma 4 26B on the LangChain Playwright toolkit, violating that assumption and explaining the 30‑point failure.  No deviation post‑registration altered the rule; all changes were “client‑side pacing” (DEVIATIONS D3, D5, D6) and cannot explain the mismatch.

## Verdict: PASS

All headline numbers in the draft post are corroborated by the analysis outputs, and the frozen‑rule logic confirms that the claim holds.  Therefore the data support the post’s claim.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
