"""Run mlx_lm.lora with Qwen3's chat template forced to enable_thinking=False.

Why: mlx-lm's mask_prompt finds the prompt length by rendering the user turn with
add_generation_prompt=True. With thinking on (the default) that render ends at
'<|im_start|>assistant\\n', so the empty '<think>\\n\\n</think>\\n\\n' block would sit in
the loss. With enable_thinking=False the prompt render equals the eval prompt
token for token, and the loss covers only the answer and its <|im_end|>. The full
training render is the same either way (checked on row 0).
"""
import sys

from transformers import PreTrainedTokenizerBase

_orig = PreTrainedTokenizerBase.apply_chat_template


def _no_think(self, *args, **kw):
    kw.setdefault("enable_thinking", False)
    return _orig(self, *args, **kw)


PreTrainedTokenizerBase.apply_chat_template = _no_think

from mlx_lm import lora  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
    lora.main()
