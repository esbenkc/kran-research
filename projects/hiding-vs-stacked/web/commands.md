## 2026-11-13 07:55:46 PST
```sh
with-model-keys ../.venv/bin/python run.py screen --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite,gemini-3.5-flash-lite,claude-haiku-4-5-20251001 --cells stacked-A --n 10 --workers 8 --budget 4.5
```
## 2026-11-13 07:57:08 PST
```sh
with-model-keys ../.venv/bin/python run.py smoke --models gpt-5.4-mini,gemini-3.5-flash-lite,gemini-3.1-flash-lite --n 2 --start 900 --workers 8 --budget 4.5
```
## 2026-11-13 07:58:40 PST (PREREG v1.2 s4: neutral-A screen)
```sh
with-model-keys ../.venv/bin/python run.py screen --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite,gemini-3.5-flash-lite --cells neutral-A --n 10 --workers 8 --budget 4.5
```
## 2026-11-13 07:59:29 PST COUNTED
```sh
with-model-keys ../.venv/bin/python run.py counted --models gpt-5.4-mini,gemini-3.5-flash-lite,gemini-3.1-flash-lite --n 40 --workers 8 --budget 4.5
```
