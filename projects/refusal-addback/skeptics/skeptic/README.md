# Vera (skeptic) — independent rebuild, oracle-1021-refusal

Own code only; inputs are the build's DATA (splits.json, acts.npz, judged jsonl, picked.json, LoRA adapter),
never the build's scripts.

| File | What it checks | Output |
|---|---|---|
| vera_splits.py | split disjointness, sources (D1), strip-set leakage (D2) | vera_splits.out |
| vera_readout.py | r_L from base FIT acts vs runs/dirs.npy; per-layer AUROC of base dir vs 50 random dirs, shuffled-label dir, own refit dir (base + stripped) | vera_readout_base.out, vera_readout_strip.out |
| vera_judge.py | my own refusal judge (claude-haiku-4-5, A/B/C letters, own rubric) + own substring scorer | used by the scripts below |
| vera_rates.py | unsteered base / stripped TEST refusal rates, Wilson CIs, judge agreement | vera_rates_unsteered.out |
| vera_pick.py | DEV pick rebuild (PREREG s6 + D7a paired budget) | vera_pick_build.out, vera_pick_build.json |
| vera_status.py | TEST status conditions 1-3 (PREREG s8 + D7), prereg parts B, C; build labels and my labels | vera_status_build.out, vera_status_myjudge.out |
| vera_reader.py | DEV-fixed 'perfect reader' on the base direction + harmful-harmless gap | vera_reader.out |
| vera_reader_null.py | the same reader on 100 random directions | vera_reader_null.out |
| vera_steer.py | own model load + own hook: finer gain grid on DEV (is the No a grid artefact?) | vera_finegrid.* |
| novelty.md | novelty check vs Lee et al. 2610.00320 App H, Arditi, Glass, 2604.27019 | — |
| signoff.md | the verdict on the title | — |

Note: my first judge prompt (one-word REFUSE/COMPLY/INCOHERENT) labelled fluent harmful compliance as REFUSE
(37 disagreements on stripped harmful). I replaced it with letter labels before any reported number; only the
letter-label numbers are used.

Paths: the scripts ran from `<workdir>/swarm-oracle-1021-refusal/skeptic/` with `ROOT = Path(__file__).parents[2]`
= `<workdir>`, where `<workdir>/kran-research` is this repo. To rerun from here, set ROOT to the repo's parent.
Judge calls go through `lib/llm.py` (cached); `with-model-keys uv run python <script>`.
