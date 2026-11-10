"""Save every critic2 (final cross-family reviewer) post of this room verbatim from the room log into this folder.
Run by Bruno (build); critic2 runs on an API model without tools in this clone, so its receipts are its posts.

    /usr/bin/python3 save_verdicts.py
"""
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
c = sqlite3.connect("file:" + str(Path.home() / "odin-context" / "rooms.db") + "?mode=ro", uri=True)
rows = c.execute("select id, ts, kind, recipient, text from messages where room = ? and sender = 'critic2' order by id",
                 ("theory-1110-cue-size-learning",)).fetchall()
for i, ts, kind, rec, text in rows:
    (HERE / f"post_{i}.md").write_text(f"# critic2 post #{i} ({ts}, kind {kind}, to {rec})\n\nSaved verbatim from the room log.\n\n{text}\n")
print(len(rows), "posts saved")
