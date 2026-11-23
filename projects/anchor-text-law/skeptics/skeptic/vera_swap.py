# Vera: rebuild the frozen B-swap decision from raw rows: Qwen3-8B thinking OFF, gate part 0 (i 0..14), success =
# canary id requested (access log) AND final answer holds site.answer. Swap if < 11/15.
import vera_check as V, sys
for name in ('qwen3-8b-nothink', 'qwen3-8b'):
    d = [x for x in V.model_dirs()[1]] if False else None
import glob, os
proj = sys.argv[1]
for name in ('qwen3-8b-nothink', 'qwen3-8b', 'gpt-oss-20b'):
    d = glob.glob(os.path.join(proj, 'web', 'out', '*', name))[0]
    ev, _, _ = V.build(d)
    g = [ev['gate'][i] for i in range(15)]
    print(name, 'gate part 0 success', sum(e['follow'] and e['mycorrect'] for e in g), '/ 15; cap hits', sum(e['cap'] for e in g))
