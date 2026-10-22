"""Vera: tokenizer ids of every statement (no special tokens), saved once to token_ids.json. Needs transformers (1021 venv)."""
import json
from transformers import AutoTokenizer
from vera_core import HERE, ROWS
tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
json.dump([tk(r["text"], add_special_tokens=False)["input_ids"] for r in ROWS], open(HERE / "token_ids.json", "w"))
