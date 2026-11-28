"""Start a command in its own session (survives the agent's shell tool ending): python harness/detach.py CMD..."""
import subprocess, sys
p = subprocess.Popen(sys.argv[1:], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
print(p.pid)
