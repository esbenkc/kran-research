# Replica of oracle.py's Figure 1 gate (FIG1_READ_PROMPT + FIG1_MATCH_PROMPT on claude-haiku-4-5), n rounds.
# Usage: with-model-keys uv run --no-project --with anthropic python gate.py <png> <post.md> [n]
import base64, json, re, sys
import anthropic

png, post = sys.argv[1], open(sys.argv[2]).read()
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
READ = """You see ONLY this image: Figure 1 of a short research blog post, shown above the post's summary. You
have not read the post. In 2 plain sentences, say (1) what question the figure asks and (2) what the answer is. If the
image does not let you say one of them, write "UNCLEAR" for it.
Answer with ONLY a JSON object: {"question": "...", "answer": "..."}"""
MATCH = """A reader saw ONLY a blog post's Figure 1 (an image) and wrote what question it asks and what the answer
is. Does the reader's question match the post's question in substance, and does the reader's answer agree with the
post's answer (yes / no / not settled, and the same direction)? Wording may differ; the substance must not.

POST TITLE (the question): {title}
POST TLDR (the answer): {tldr}
READER'S QUESTION: {q}
READER'S ANSWER: {a}

Answer with ONLY a JSON object: {{"pass": true or false, "why": "one sentence"}}"""
title = re.search(r"^title:\s*(.*)$", post, re.M)[1].strip().strip("\"'")
tldr = re.search(r"^\W*TLDR\W*\s*(.*)$", post, re.M | re.I)[1][:600]
js = lambda t: json.loads(re.search(r"\{.*\}", t, re.S)[0])
c = anthropic.Anthropic()
img = base64.b64encode(open(png, "rb").read()).decode()
ok = 0
for i in range(n):
    r = c.messages.create(model="claude-haiku-4-5", max_tokens=400, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}, {"type": "text", "text": READ}]}])
    qa = js(r.content[0].text)
    m = c.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=[{"role": "user", "content":
        MATCH.format(title=title, tldr=tldr, q=qa["question"], a=qa["answer"])}])
    v = js(m.content[0].text)
    ok += bool(v["pass"])
    print(f"{i}: {'PASS' if v['pass'] else 'FAIL'} | Q: {qa['question']} | A: {qa['answer']} | {v['why']}")
print(f"gate {ok}/{n}")
