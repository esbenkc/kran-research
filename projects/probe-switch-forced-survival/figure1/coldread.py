# /// script
# dependencies = ["anthropic"]
# ///
"""Cold-reader check (Owain): a cheap model sees ONLY the Figure 1 PNG and states the question and the answer.
Usage: with-model-keys uv run coldread.py <png> [n]"""
import base64
import sys

import anthropic

png, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
img = base64.b64encode(open(png, "rb").read()).decode()
c = anthropic.Anthropic()
for i in range(n):
    m = c.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
        {"type": "text", "text": "This image is Figure 1 of a research blog post. In exactly 2 sentences: "
                                 "(1) what question does the post ask, and (2) what is its answer?"}]}])
    print(f"--- {i + 1}:", m.content[0].text.strip())
