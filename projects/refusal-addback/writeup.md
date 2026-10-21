# Does adding back the refusal direction at one layer restore a stripped model's refusals without over-refusing?

**Question:** after a LoRA fine-tune strips refusal from Qwen2.5-1.5B-Instruct, does adding the base model's
refusal direction (difference of means, Arditi et al. arXiv 2406.11717) back at one layer restore refusal of
harmful prompts without new refusals of harmless prompts?
**Sources:** prompts from github.com/andyrdt/refusal_direction (dataset/splits); strip pairs from
huggingface.co/datasets/LLM-LAT/harmful-dataset; XSTest-safe prompts (Röttger et al. arXiv 2308.01263) via
huggingface.co/datasets/natolambert/xstest-v2-copy. Model Qwen/Qwen2.5-1.5B-Instruct. Judge gpt-4.1-mini.
**Method:** PREREG.md (frozen 8dcf3af) + DEVIATIONS.md. Everything local on a Mac mini M4 (torch MPS, bf16).

## Rebuild

Run from this folder with `PY=../../.venv/bin/python`:

| step | command | output |
|---|---|---|
| raw data | `./fetch_data.sh` | data/raw/ (not committed) |
| splits | `$PY prep_data.py` | data/splits.json |
| seed 0 | `./chain_s0.sh` | runs/dirs.npy, runs/base/, runs/strip_s0/ (adapter, readout, unsteered TEST, DEV sweep) |
| pick + TEST + seeds 1-2 + base DEV + XSTest | `./chain_test.sh` | runs/picked.json, runs/*/test.jsonl, runs/*/xstest.jsonl |
| judge | `with-model-keys $PY judge.py <file.jsonl>` (judge_loop.sh runs it on every file) | `<file>.judged.jsonl` |
| numbers | `$PY results.py` | runs/results.json, figures/refusal-addback/figure_data.json |
| hand-label export | `$PY export_arms.py` | runs/test_arms.jsonl |

Files (all committed except adapters and acts.npz, which stay on the machine):
- `runs/dirs.npy` [28, 1536]: base r_L = mean(FIT harmful) - mean(FIT harmless), last prompt token, input of block L.
- `runs/<model>/readout.json`: per-prompt projections on unit r_L for FIT/DEV/TEST, TEST AUROC per layer,
  AUROC of 3 fixed random unit directions per layer; `refit_dirs.npy` = the model's own diff of means on FIT.
- `runs/<model>/{dev,test,xstest}.jsonl`: one row per generation: layer (null = no add-back), gain k, dir
  (none | base | rand0-2 | refit | base_harmless_only), kind, i (index into data/splits.json), split, src,
  prompt, text, str_ref (string-match scorer). `.judged.jsonl` adds `judge`.
- `runs/picked.json`: k*_L per layer, L*, L_mid, all DEV counts (harmful refused of 64, new harmless refusals).
- `logs/footprint.log`: peak footprint and rss per model process.

## Findings

(filled in after TEST)

## Blog post

Slug: `blog.kran.ai/_notes/refusal-addback.md`, figure `refusal-addback`.
