# critic2 command at 2026-11-23T10:30:54.464Z (verbatim from the session)
python3 -c 'import subprocess
commit="342b04d7"
checks=[("assets/figures/anchor-text-law/data.json",18),("_notes/anchor-text-law.md",6)]
for path, ln in checks:
    data=subprocess.check_output(["git","-C","blog.kran.ai","show",f"{commit}:{path}"], text=True)
    line=data.splitlines()[ln-1]
    print(f"{path}:{ln}:{line}")'
