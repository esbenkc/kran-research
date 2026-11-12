
## 2026-11-11 23:07:23 PST counted key at e9b97fde8 (PREREG v1.1 8b155cc89)

```sh
cd key && with-model-keys ../.venv/bin/python run.py counted --seeds 20 --workers 8 --budget 4.50   # logs/counted.log
```

## 2026-11-11 23:16:12 PST counted key RELAUNCHED (the 23:09 nohup job died with the shell, 0 raw files; its cached calls replay as the same samples)

```sh
cd key && with-model-keys ../.venv/bin/python run.py counted --seeds 20 --workers 8 --budget 4.50   # logs/counted.log (omp service ww-key-counted)
```

## 2026-11-11 23:29:53 PST counted web (winner = full by analysis.json web_winner, PREREG s5)

```sh
cd web && with-model-keys ../.venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --cells neutral-cued,full-cued --n-cued 30 --start 0 --workers 8 --budget 4.75   # logs/counted.log
```
