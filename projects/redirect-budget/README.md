# redirect-budget

Does a bound computed before training predict which attention heads a learned switch hijacks? Qwen3-1.7B and
Qwen3-0.6B, PyTorch 2.14.1 (MPS bf16 forwards, CPU fp32 / float64 numpy for the bound), transformers 5.20.2, Mac mini
M4. All local; no model-API calls ($0). Frozen plan: `PREREG.md` (v1 2e3ab198, v1.1 57fa6613, v1.2 a647b207, all before
any carrier, bound or measured shift). Derivation and equation checks: `derivation.md`, `derive_check.py`.
Changes with times: `DEVIATIONS.md`. Proof matrix: `proof-matrix.md`. Answer: **No** (status REJECTED, H = 0 of 5
runs with a clear hijacked head).

## The 7 saved runs (no new training except the forward test)

- `src1027/`: copy of `probe-switch-frozen-heads` (branch swarm/oracle-1027-forced-attention). MLP-only LoRA on blocks
  20-27 of Qwen3-1.7B with every attention weight frozen, seeds 0-2; cue = a read-only probe hook (last-bit leak).
  Inputs: the block-20 cache `src1027/raw/cache/{none,probe}_A_b20.npy`, rebuilt by `src1027/cache.py --pass A`;
  sha256 matches `src1027/data/manifest_raw.json` bit for bit (`data/gates.json`).
- `src1101/`: copy of `sink-value` (branch swarm/oracle-1101-sink-value). Qwen3-0.6B blocks 26-27 trained by reward
  (grader switch s0/s1, monitor switch s0/s1), merged bf16; cached block-26 inputs, cue on/off.

## What runs (in order; each step's outputs are committed before the next reads them)

1. `carriers17.py --run mlp8_none_s{0,1,2}` and `carriers06.py`: PREREG sec 2. Swap one head's whole output (its
   128-column slice of the o_proj input) from the cue-off run into the cue-on run at all positions; informed share
   removed; carrier = largest share (1.7B dev split, 64 prompts; 0.6B the 200 prompts). `make_carriers.py` applies
   the 'undefined' clause -> `data/carriers.json` (commit 983ef619, before bounds.py existed).
2. `capture.py`: per model state, both cue conditions, each block's bf16 input from the trained tail, then the
   attention internals at the answer token recomputed in fp32 (q and k after QK-norm and RoPE, raw q/k norms,
   clean attention, write norms). Check: p.V vs the block's own sdpa output, max rel err 0.019-0.028
   (`data/capture/*.json`). Big arrays in `raw/capture/` (local, sha256 in the json).
3. `bounds.py`: rows S, Q, K, C (measured model's weights and write sizes), RANK (untrained weights, untrained cue-off
   forward, write size 1), RANK0, RANK_Q, RANK_C, PROXY; softmax step `tv_bound` equal to `derive_check.T_sound`
   (selfcheck). -> `data/bounds/<run>.npz` (head x prompt arrays), `data/bounds.json` (commit cb4e3efe, sha posted
   in the room before compare.py read any measured shift).
4. `compare.py`: measured score change (max_j dz_j - min_j dz_j at the answer token, cue on minus off) and TV per head;
   scout clauses C1-C4 for every row; the RANK status test with the frozen baselines; carrier places under every
   ranking; headroom; Spearman. -> `data/summary.json`, `data/measured/<run>.npz`.
5. `proof_matrix.py` -> `proof-matrix.md`; `figure_data.py` -> `figures/redirect-budget/data.json` (+ blog copy).
6. Forward test (PREREG sec 8): `chain_j4.sh` trains mlp8 s3 (and s4) with the unchanged `src1027/train.py`, then
   runs the carrier sweep.

## Rebuild

```sh
cd ~/work/theory-1101-redirect-budget; PY=env/.venv/bin/python; P=kran-research/projects/redirect-budget
(cd $P/src1027 && ../../../../$PY cache.py --pass A)          # 3 min, ~5 GB; check sha256 vs data/manifest_raw.json
for r in mlp8_none_s0 mlp8_none_s1 mlp8_none_s2; do $PY $P/carriers17.py --run $r; done   # ~6 min, 3.6 GB each
$PY $P/carriers06.py && python3 $P/make_carriers.py          # 1 min, 0.8 GB
$P/chain_j3.sh                                                # captures, ~1.5 min total, <= 3.5 GB
$PY $P/bounds.py                                              # 3 min CPU, 2.1 GB
$PY $P/compare.py && python3 $P/proof_matrix.py && python3 $P/figure_data.py
```
Offline (from the committed data only): `compare.py` needs `raw/capture/` for the measured shifts; everything after it
(`proof_matrix.py`, `figure_data.py`) runs from `data/`. `data/measured/<run>.npz` holds the measured per head x prompt
arrays, so the headline cells can be rebuilt without any model: `data/bounds/*.npz` + `data/measured/*.npz` +
`data/carriers.json`.
