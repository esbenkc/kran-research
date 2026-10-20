# megaprobe-spin-lies raw data

Room oracle-1020-megaprobe-spin-lies. All files line up by `idx` (row number 0-999) within a policy, and every policy
answers the same 1000 prompts in the same order (rows 0-399 = refit-probe's TEST prompts, `split == "test"`).
Policies: `base` (untrained Qwen2.5-0.5B-Instruct), `lam1_s0`, `lam1_s1` (DPO against one probe), `ens_s0`, `ens_s1`
(DPO against the max over 8 probes). No new generation and no new judge calls in this run.

| file | written by | content |
|---|---|---|
| `refit_probe/answers_<p>.jsonl`, `refit_probe/labels_<p>.jsonl` | copied unchanged from oracle-1019-refit-probe | answers and labels: `y_judge` (gemini-3-flash-preview, 1 = DECEPTIVE), `y_string` (0 = the fact's key words are in the answer). Spin lie = `y_judge 1 & y_string 0`. base has 1600 rows; only rows 0-999 are used |
| `cache/<p>_<read>/` (NOT committed, 22 GB) | cache.py | `tok.npy` (n_tok, 24, 896) fp16: every answer token, block outputs 1-24; `offsets.npy`; `follow.npy` (n, 17, 24, 896): the follow-up turn tokens, last = "Yes"; `meta.json`. read `own` = the policy reads its answers, `base` = the untrained model reads them (held text; meta's `adapter` names the policy whose answers were read, the weights are the base model's) |
| `folds.json` | probes.py | prompt_id -> fold (0-4), seed 1020, shared by every model and arm |
| `draws.json` | probes.py | policy -> fold -> 5 draws of 400 row indices (label set "all") |
| `hp.json`, `hp_c2.json` | probes.py pilot / pilot_ext, fit_c2.py pilot | hyperparameter grids and the chosen values (base model, outer fold 0 training rows only) |
| `fits/<p>_<read>.json` | probes.py fit | per arm: a (1000 x 5) matrix of out-of-fold scores, one column per draw (null where a draw was not run); `info` = Atlas layers kept + train CE, megaprobe best epoch + inner val AUROC, draw class sizes |
| `fits/<p>_own_c2.json` | fit_c2.py fit | arm c2 (FIBS axial) scores, same layout |
| `fits/pooled_fact.json` | probes.py pooled | fact-present fit pooled over the 4 trained models; rows policy-major (row j = policy j // 1000) |
| `results/gate.json` | gate.py | PREREG s4 gate: re-forward vs refit-probe acts, cosine per read |
| `results/summary.json`, `results/verdict.json` | score.py | every AUROC with 95% paired prompt-bootstrap CI (B = 1000, seed 1020), the PREREG s6 verdict (+ D1, D5), the s7 prediction outcome, agent-label subclasses |
| `results/oof_scores.jsonl` | score.py | one row per policy x read x answer: labels + the OOF score of every arm (mean over its draws) |
| `results/figure_data.json` | score.py | the figure's data (blog assets/figures/megaprobe-spin-lies/data.json) |
| `logs/` | all | run logs and footprint logs (memory cap) |

Rebuild from kran-research/ (torch MPS, Mac mini M4):

    uv run python projects/megaprobe-spin-lies/cache.py          # ~23 min, 22 GB to data/cache/
    uv run python projects/megaprobe-spin-lies/gate.py
    uv run python projects/megaprobe-spin-lies/probes.py pilot && uv run python projects/megaprobe-spin-lies/probes.py pilot_ext
    uv run python projects/megaprobe-spin-lies/probes.py fit     # ~60 min
    uv run python projects/megaprobe-spin-lies/fit_c2.py pilot && uv run python projects/megaprobe-spin-lies/fit_c2.py fit
    uv run python projects/megaprobe-spin-lies/probes.py pooled
    uv run python projects/megaprobe-spin-lies/score.py          # CPU, ~1 min
