from pathlib import Path
p=Path('athanorscience.com/_notes/route-first-law.md')
for i,l in enumerate(p.read_text().splitlines(),1):
    if 'It is clear to me' in l or 'With no route' in l or 'With a correct route' in l: print(i, l)
