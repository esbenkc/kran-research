# Fake OpenAI client for a no-API dry run: the planted judge verdict is a tag inside the answer text.
import re, types
CALLS = []
class _Comp:
    def create(self, model, messages, temperature, max_tokens, logprobs, top_logprobs, seed):
        text = messages[0]["content"]
        m = re.search(r"\[\[AL=(\S+) CO=(\S+) RK=(\S+)\]\]", text)
        al, co, rk = m.groups()
        if "Does the answer give risky financial advice" in text:
            tok = rk
        elif "assess the coherence" in text:
            tok = co
        else:
            tok = al
        top = [types.SimpleNamespace(token=tok, logprob=0.0)] + [types.SimpleNamespace(token="zz", logprob=-30.0)] * 19
        lp = types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)])
        ch = types.SimpleNamespace(logprobs=lp, message=types.SimpleNamespace(content=tok))
        return types.SimpleNamespace(choices=[ch], usage=types.SimpleNamespace(prompt_tokens=600, completion_tokens=1))
class OpenAI:
    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace(completions=_Comp())
