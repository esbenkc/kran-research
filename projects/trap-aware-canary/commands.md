## 2026-11-09 23:00:31 PST smoke at 29d56217

```sh
cd web && with-model-keys ../.venv/bin/python run.py smoke --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --n-cued 2 --n-free 2 --n-side 2 --start 900 --workers 8 --budget 0.5
cd key && with-model-keys ../.venv/bin/python run.py smoke --seeds 2 --workers 8 --budget 0.3
```

## 2026-11-09 23:04:20 PST counted web A+C at f0045e33 (PREREG v1.1 f0045e33)

```sh
cd web && with-model-keys ../.venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --cells neutral-cued,logging-cued,warning-cued,rule-cued,neutral-free,warning-free,rule-free --n-cued 60 --n-free 30 --workers 8 --budget 2.2
cd key && with-model-keys ../.venv/bin/python run.py counted --seeds 30 --workers 8 --budget 2.3
```

## 2026-11-09 23:06:29 PST web lane RESTARTED (all 990 jobs failed at once: NameError sys from 8 threads importing the SDKs; run.py now pre-imports them; 0 counted web runs existed)

```sh
cd web && with-model-keys ../.venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --cells neutral-cued,logging-cued,warning-cued,rule-cued,neutral-free,warning-free,rule-free --n-cued 60 --n-free 30 --workers 8 --budget 2.2   # logs/counted2.log
```

## 2026-11-09 23:15:30 PST block D (spend after A/B/C = web $1.987 + key $1.951 = $3.938 <= $4.25, DEVIATIONS D1)

```sh
cd web && with-model-keys ../.venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --cells marked-cued --n-side 40 --workers 8 --budget 2.45   # logs/counted_D.log
```

