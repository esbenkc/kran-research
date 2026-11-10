## 2026-11-09 23:00:31 PST smoke at 29d56217

```sh
cd web && with-model-keys ../.venv/bin/python run.py smoke --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --n-cued 2 --n-free 2 --n-side 2 --start 900 --workers 8 --budget 0.5
cd key && with-model-keys ../.venv/bin/python run.py smoke --seeds 2 --workers 8 --budget 0.3
```

