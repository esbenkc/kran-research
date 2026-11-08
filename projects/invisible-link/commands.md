## 2026-11-07 23:06:22 PST pilot at 40c0950e
with-model-keys .venv/bin/python run.py pilot --plan pilot --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite,claude-haiku-4-5-20251001 --workers 8 --budget 4.5

## 2026-11-07 23:11:39 PST counted at 64674500

```sh
with-model-keys .venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite,claude-haiku-4-5-20251001 --cells stuck-raw-hidden_fit,stuck-md-hidden_fit,stuck-raw-hidden_nonfit,stuck-md-hidden_nonfit,stuck-raw-visible_fit,stuck-md-visible_fit,stuck-raw-nolink,stuck-a11y-hidden_fit,stuck-a11y-hidden_nonfit,stuck-a11y-sronly_fit,rstuck-raw-hidden_fit,rstuck-md-hidden_fit,rstuck-raw-hidden_nonfit,rstuck-md-hidden_nonfit,rfree-raw-hidden_fit,rfree-md-hidden_fit,rfree-raw-hidden_nonfit,rfree-md-hidden_nonfit --n-title 30 --n-stuck-visible 30 --n-ctl 10 --n-route 15 --workers 8 --budget 4.90
```

## 2026-11-07 23:13:07 PST crawler check at 1aab7384

```sh
.venv/bin/python crawler.py --n 5 --phase crawler_counted && .venv/bin/python score.py --phase crawler_counted
```

## 2026-11-07 23:13:50 PST counted RESUMED without Haiku (cost rule s6, re-projection \$5.75 > \$4.50) at 1aab7384

```sh
with-model-keys .venv/bin/python run.py counted --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite --cells stuck-raw-hidden_fit,stuck-md-hidden_fit,stuck-raw-hidden_nonfit,stuck-md-hidden_nonfit,stuck-raw-visible_fit,stuck-md-visible_fit,stuck-raw-nolink,stuck-a11y-hidden_fit,stuck-a11y-hidden_nonfit,stuck-a11y-sronly_fit,rstuck-raw-hidden_fit,rstuck-md-hidden_fit,rstuck-raw-hidden_nonfit,rstuck-md-hidden_nonfit,rfree-raw-hidden_fit,rfree-md-hidden_fit,rfree-raw-hidden_nonfit,rfree-md-hidden_nonfit --n-title 30 --n-stuck-visible 30 --n-ctl 10 --n-route 15 --workers 8 --budget 4.90
```
