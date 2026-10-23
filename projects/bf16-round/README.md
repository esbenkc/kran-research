# bf16-round

Does an explicit 16-bit round at the probe's split point make a compiled read-only probe invisible to a trained
detector? Qwen3-1.7B, bf16, PyTorch 2.14.1 (MPS, `torch.compile` / Inductor Metal backend), Mac mini M4 (16 GB).
Follow-up to `projects/probe-sensing-mps` (same prompts, same probe, its 3 saved detectors).
Frozen plan: `PREREG.md`; changes with commit times: `DEVIATIONS.md`. All local, no model-API calls.

## Probe and fixes

Probe = forward hook on block 14 `self_attn.q_norm` that keeps a reference to the output. Every fix sits in
q_norm's forward (`bits.py make_forward`), so it is in BOTH the probed and the unprobed compiled graph.

| fix id | q_norm returns | figure label |
|---|---|---|
| `none` | stock `weight * bf16(h)` | Leaking build (no fix) |
| `cast` | `stock.float().to(bf16)` | Naive cast |
| `bitcast` | `stock.view(int16).view(bf16)` | Bit-view round |
| `round16` | `(weight.float() * bf16(h).float()).to(bf16)` (title arm) | 16-bit round |
| `intround` | integer round-to-nearest-even of the same fp32 product, then `.to(bf16)` | (text only) |
| `round16f` | `round16`, plus the query rotation (rope) in fp32 and one bf16 round after it (D1) | 16-bit round + 32-bit next step |
| `barrier` | stock through an opaque custom op (clone) | Opaque split |
| emu1 `none` | stock, `torch._inductor.config.emulate_precision_casts = True` | Precision-emulation setting |

## Rebuild (in order; one heavy job at a time, footprint cap 5 GB via `watchdog.py`)

The venv is an APFS clone of probe-sensing-mps's (`../../.venv` -> `bf16env/.venv`, torch 2.14.1, transformers 5.20.2).
Detectors: `bf16env/detectors/*.pt` = `cp -c` of probe-sensing-mps `raw/step2/Qwen3-1.7B/runs/*.pt`.

```sh
.venv/bin/python projects/bf16-round/bits.py --emulate 0 --code-only --n 1 --copies A --out projects/bf16-round/raw/codegate   # Metal gate
.venv/bin/python projects/bf16-round/bits.py --emulate 0                                       # chain_t2.sh
.venv/bin/python projects/bf16-round/bits.py --emulate 1 --fixes none,cast,round16
.venv/bin/python projects/bf16-round/bits.py --emulate 0 --backend eager --fixes none,round16 --copies A
.venv/bin/python projects/bf16-round/bits.py --emulate 0 --fixes none,round16f --out projects/bf16-round/raw/bits_r2f   # chain_t2b.sh
.venv/bin/python projects/bf16-round/timing.py
.venv/bin/python projects/bf16-round/cache_tail.py --fix round16                              # chain_t3.sh (T4)
.venv/bin/python projects/bf16-round/tail_1022.py --backend round16 --build probe --null none --seed 0
.venv/bin/python projects/bf16-round/eval_detectors.py --emulate 0 --fixes none,round16 --runs compile_probe_none_s0,compile_probe_none_s1,compile_probe_none_s2,compile_probe_never_s0,fresh_round16_probe_none_s0
.venv/bin/python projects/bf16-round/eval_detectors.py --emulate 1 --fixes none --runs compile_probe_none_s0,compile_probe_none_s1,compile_probe_none_s2
.venv/bin/python projects/bf16-round/heads.py
<nightly venv>/bin/python projects/bf16-round/bits.py --emulate 0 --fixes none,round16 --out projects/bf16-round/raw/bits_nightly   # chain_t4.sh (D2)
.venv/bin/python projects/bf16-round/table.py                                                  # data/bits_table.json
.venv/bin/python projects/bf16-round/make_figure.py --publish
```

## Files

- `raw/bits*/Qwen3-1.7B/<backend>_emu<0|1>/` (not in git, ~30 GB): per graph `<fix>_<noprobe|probe>_<A|B>`:
  `_last.npy` uint16 [pass, prompt, vocab] answer-position logit bits (pass 0 = warm-up); `_hidden.npy` uint16
  [prompt, 71, 2048] final-norm output, pass 1; `_sha.json` [pass][prompt][position] SHA256 of each logit row;
  `_A_alllogits_pass1.npy` uint16 [prompt, 71, vocab]; `code/` generated kernel source; `meta.json`.
- `raw/detectors/Qwen3-1.7B/emu<e>/<fix>__<run>.json`: per case top token, Yes/No log-probs, SHA256 of the
  all-position logits; `_last.npy` uint16 [prompt, probe, vocab].
- `raw/step2/Qwen3-1.7B/round16_C/`: cached block-26 inputs for T4; `raw/step2/Qwen3-1.7B/runs/`: fresh detector.
- `raw/timing/timing.json`: per-forward ms, medians, cost and bootstrap interval.
- `data/` (in git): `bits_table.json`, `heads.json`, detector summaries.
