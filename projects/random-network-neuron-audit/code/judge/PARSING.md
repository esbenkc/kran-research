# Parser change after the first A+C pass (2026-10-16 ~00:15 CPH, before any rate was computed)

The frozen rubric, prompts, models, temperature and verdict rule did not change. Only how the judges' raw text is read changed.

First pass (parser v1, raw kept in `data/judgments_parser_v1.jsonl`): 3 of 600 label answers and 9 of 600 detection answers could not be read.

Causes, from the raw text:
1. claude-haiku-4-5 sometimes printed a JSON answer, then "Wait, let me reconsider", then a second answer. v1 took everything from the first `{` to the last `}`, which is not valid JSON. **Fix:** take the last complete JSON object that has the expected key (the judge's final answer).
2. gpt-4.1-mini sometimes returned fewer than 5 detection picks, also after the format reminder. **Fix:** strict parse first (exactly 5); only if the reminder retry also fails, accept 0-5 distinct picks. Hits are counted over the picks given, so a short list can only lower the score.
3. gpt-4.1-mini sometimes listed job counts that add up to more than 20 (double counting), also after the reminder. **Fix:** strict parse first (sum = 20); only if the reminder retry also fails, keep the answer and set `count_overflow`. Only `main_count` enters the verdict. Dropping these units would have removed exactly the messy ones.

After the fix (A+C): 0 unreadable answers; 3 labels flagged `count_overflow`; 5 detection answers with fewer than 5 picks. Extra API cost of the re-run: under $0.01 (everything else came from the cache).

No arm rate, contrast or hypothesis flag was computed before this change.
