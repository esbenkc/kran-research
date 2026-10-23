"""Cold read of Figure 1: a cheap model sees ONLY the PNG and states the post's question and answer.
Run: with-model-keys python3 coldread.py <png> [n]"""
import base64, json, os, sys, urllib.request

PROMPT = ("This image is the first figure of a short research blog post. Using only the image, state in exactly "
          "2 sentences: (1) the question the post asks, (2) the answer it finds.")
png, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
b64 = base64.b64encode(open(png, "rb").read()).decode()


def post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers})
    return json.load(urllib.request.urlopen(req, timeout=120))


def haiku():
    r = post("https://api.anthropic.com/v1/messages",
             {"model": "claude-haiku-4-5", "max_tokens": 300, "messages": [{"role": "user", "content": [
                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                 {"type": "text", "text": PROMPT}]}]},
             {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    return r["content"][0]["text"]


def flash():
    key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={key}",
             {"contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": b64}}, {"text": PROMPT}]}]},
             {})
    return r["candidates"][0]["content"]["parts"][-1]["text"]


for name, fn in (("haiku", haiku), ("flash", flash)):
    for i in range(n):
        print(f"--- {name} #{i + 1}\n{fn().strip()}\n", flush=True)
