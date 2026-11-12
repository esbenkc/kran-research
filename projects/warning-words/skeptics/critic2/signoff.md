# critic2 sign-off: defense-1112-warning-words

Reviewer: Kasper (critic2), final cross-family review. Rebuild run on openai/gpt-5.5 (verdict #20231, room defense-1112-warning-words); this file and the re-run of rebuild.py were written by the Claude (opus) fallback after the GPT-5.5 session hit its $2.50 cap.

## Answer to the question
UNCONFIRMED (Not settled). The pre-registered prediction failed.

## Numbers rebuilt from raw data (own code: rebuild.py, output: rebuild_output.json)
- Input: key/raw (600 JSONL runs: 5 agents x 6 arms x 20 seeds) and web/results/scores_counted.jsonl (180 runs).
- Agent-equal decoy-key use: neutral 0.93, trap 0.78, honeypot 0.62, name it 0.63, full 0.31, verbalise 0.93.
- Gap G = cut(name it) - cut(honeypot) = -0.01, 95% bootstrap CI [-0.13, 0.11] (20,000 resamples, seed 552).
- Full-warning cut 0.62 CI [0.53, 0.71]; interaction -0.01 CI [-0.16, 0.14].
- Web hidden-link follows: neutral 0.9444, full 0.8; cut 0.1444 CI [0.0556, 0.2333].

## Why Not settled
G is near zero, but the 95% CI reaches +0.11; the frozen No rule needed the upper end below +0.10. Yes needed G >= +0.20 with CI-low > 0; G is -0.01.

## Preview checked
https://deploy-preview-197--kranai.netlify.app/warning-words/ at blog commit 8c50e17: text matches these numbers; Figure 1 answer line gives the reason for Not settled.

Verdict: SIGN-OFF on UNCONFIRMED.
