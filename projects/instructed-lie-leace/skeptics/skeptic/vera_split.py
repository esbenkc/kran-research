"""R2 at the cap: blind-panel catch split by what the model now says on lie prompts (own gens, own panel scores)."""
import json, re
import numpy as np
P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data/"
items = json.load(open(P + "ref/items_spread.json"))


def name(g):
    m = re.match(r"\s*((?:the )?[A-Z][\w'’ .-]*?)(?:\.| \(|,|\n|$)", g)
    return m.group(1).strip() if m else None


def norm(c):
    return c.replace("the ", "", 1) if c and c.startswith("the ") else c


g = json.load(open("ext_R2_s300_gens.json"))
idx, pan, line = np.load("vera_pan_R2_s300.npy")
line = line[0]; sc = dict(zip(idx.astype(int), pan))
truth = [k for k, t in g.items() if items[int(k)]["lie"] == 1 and name(t) and norm(name(t)) == norm(items[int(k)]["country"])]
false = [k for k, t in g.items() if items[int(k)]["lie"] == 1 and k not in truth]
tr = [k for k in g if items[int(k)]["lie"] == 0]
f = lambda ks: (int(sum(sc[int(k)] > line for k in ks)), len(ks))
print("R2 s300 blind panel caught (caught, n): lie prompts now naming the TRUE country %s, lie prompts not %s, truthful prompts %s" % (f(truth), f(false), f(tr)))
