# critic2 command at 2026-11-23T10:24:40.148Z (verbatim from the session)
python3 -c "import json; d=json.load(open('kran-research/projects/anchor-text-law/web/out/B/qwen3-8b/raw/anchor-none/000.json')); print(d['site']); print(d['actions']); print('visible_followed',d.get('visible_followed'),'correct',d.get('correct'),'exposed',d.get('exposed'))"
