import base64, sys, os
import anthropic
from google import genai
from google.genai import types

png = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
data = open(png, "rb").read()
Q = ("This image is Figure 1 of a research blog post. Looking only at the image, state the post's "
     "question and its answer in exactly 2 sentences.")

a = anthropic.Anthropic()
for i in range(n):
    r = a.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(data).decode()}},
        {"type": "text", "text": Q}]}])
    print(f"haiku {i}:", r.content[0].text.strip().replace("\n", " "))

g = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
for i in range(n):
    r = g.models.generate_content(model="gemini-flash-latest", contents=[types.Part.from_bytes(data=data, mime_type="image/png"), Q],
                                  config=types.GenerateContentConfig(temperature=1.0))
    print(f"gemini {i}:", r.text.strip().replace("\n", " "))
