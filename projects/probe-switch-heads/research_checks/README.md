# research_checks (Iris, research)

Mechanism checks on TRAINING prompts only (never dev or test), run before the dev-picked head scores existed.
CPU, float32 compute on the bf16 weights and the bf16 cached block-26 inputs. Peak footprint 1.65-1.76 GB
(`/usr/bin/time -l`).

| File | What |
|---|---|
| `head_attention.py` | For the untrained tail and each trained tail (blocks 26-27 + final norm): per query head, attention from the answer position with and without the probe, the change of the head's o_proj-input slice, a linear direct-path split of the answer-position change (block-26 input, attn26, mlp26, attn27, mlp27), and optional cut patches (`--cut`: the head's slice from the unprobed run, all positions). |
| `head_attention.json` | The 3 saved detectors of the probe-sensing-mps run (Yes/No question, full blocks 26-27), 64 train prompts. |
| `head_attention_L.json` | The LoRA runs of this post (mountains/sea question), 64 train prompts, with `--cut`. |

Inputs (not copied; regenerate with this project's `step1_torch.py` cache step, or the earlier run's files):
- this post: `raw/step2/Qwen3-1.7B/compile_C/{meta.json,none_rec_tail.npy,probe_rec_tail.npy}`, `data/prompts.jsonl`,
  `raw/step2/Qwen3-1.7B/runs/lora_none_s*_merged.pt`
- saved detectors: probe-sensing-mps `raw/step2/Qwen3-1.7B/compile_C/*` and `runs/compile_probe_none_s{0,1,2}.pt`

Commands (inputs placed in `in/` and `inL/` next to the script):

    python head_attention.py --n 64
    python head_attention.py --n 64 --in-dir inL --answers A,B \
        --models L_s0=inL/lora_none_s0_2048_merged.pt,L_s1=inL/lora_none_s1_merged.pt \
        --cut "27.7;27.12;27.3;27.1;27.7+27.12;27.7+27.12+27.3" --out head_attention_L.json

Key numbers (train prompts): detectors' head 27.1 moves 0.24 / 0.29 / 0.24 of its attention from `<|im_end|>` to the
blank line after `<think>`; LoRA runs' head 27.7 moves 0.68 (run 1) and 0.81 (run 2) between the same two tokens.
Cutting 27.7 alone removes 0.735 and 0.917 of the probe's push; MLP27 writes the answer (direct push +39.3 and +40.1
of 38.0 and 41.0 nats).
