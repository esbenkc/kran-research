# critic2 command at 2026-11-23T10:26:01.342Z (verbatim from the session)
python3 -c 'files=[("blog.kran.ai/_notes/anchor-text-law.md",range(1,8)),("blog.kran.ai/assets/figures/anchor-text-law/data.json",range(16,20)),("kran-research/projects/anchor-text-law/results/analysis.json",range(174,202))];
for f,rs in files:
 print("--",f)
 for i,line in enumerate(open(f),1):
  if i in rs: print(f"{i}:"+line.rstrip())'
