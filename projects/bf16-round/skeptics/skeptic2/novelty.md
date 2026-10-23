# skeptic2 novelty check

Checked by Hugo from public sources, independent of the experiment code.

Sources opened:
- Sauers `watching` README: https://raw.githubusercontent.com/SauersML/watching/main/README.md
- Sauers mechanism audit: https://raw.githubusercontent.com/SauersML/watching/main/docs/mechanism-audit.md
- Hugging Face model card: https://huggingface.co/Sauers/watching-probe-detector
- Web search query: `SauersML watching probe bfloat16 round emulate_precision_casts explicit round split point detector compiled probe`.

Findings:
- Sauers' public result trains Qwen3-8B to detect a read-only activation probe under `torch.compile`; the public model card reports 128/128 fresh L40S answers and says the detector reads a numerical difference from a layer-14 hook.
- Sauers' README has controls for no probe, two kernels, copy probe, hidden-state tools, training/deployment, GPU transfer, and circuit/head tracing. I found no reported test where an explicit bfloat16 round at the probe split point is inserted in both probed and unprobed graphs and then checked bit-for-bit.
- Sauers' mechanism audit says the CUDA buffer/capture footprint includes a changed query-normalization reduction layout and that the native rotary path still uses an FP32 intermediate before the BF16 destination. That source does not claim that a split-point bf16 round fixes the leak; it argues the causal question remains open.
- Search results surfaced PyTorch/XPU precision-cast issues and Sauers model cards, not a public experiment matching this room's title arm or `emulate_precision_casts` fix test.

Novelty verdict for this room: I found prior public evidence that compiled read-only probes can be detected, but no public source that tests the specific fix: an explicit 16-bit round at the split point, measured by warm bit equality and detector transfer on this MPS stack.
