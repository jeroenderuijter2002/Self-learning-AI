"""
Exporteert echte GPT-2-data voor de interactieve pagina's in interactief/.
    python tools/export_interactive_data.py
Schrijft interactief/data/*.js (als JS-variabelen, zodat de pagina's ook werken als je ze lokaal opent).
"""
import json, numpy as np, torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = Path(__file__).resolve().parents[1] / "interactief" / "data"
OUT.mkdir(parents=True, exist_ok=True)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
show = lambda i: tok.decode([i])

def write(name, var, obj):
    (OUT / f"{name}.js").write_text(f"window.{var} = " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + ";\n", encoding="utf-8")
    print(name, round((OUT / f"{name}.js").stat().st_size / 1024), "KB")

# ---------- attention: alle 12x12 heads, onderdriehoek als ints 0..100
att_sentences = [
    "When Mary and John went to the store, John gave a drink to",
    "The cat sat on the mat because it was tired.",
    "The keys to the cabinet are on the table.",
    "Harry Potter studied at Hogwarts. Harry Potter was a wizard.",
]
att = []
for s in att_sentences:
    ids = tok(s, return_tensors="pt").input_ids
    with torch.no_grad():
        A = torch.stack(model(ids, output_attentions=True).attentions)[:, 0].numpy()  # L,H,T,T
    T = A.shape[-1]
    tri = [[[ [int(round(A[l, h, i, j] * 100)) for j in range(i + 1)] for i in range(T)] for h in range(12)] for l in range(12)]
    att.append({"text": s, "tokens": [show(t) for t in ids[0].tolist()], "A": tri})
write("attention", "ATTN_DATA", att)

# ---------- sampling: top-300 logits + histogram van de rest
prompts = [
    "The capital of the Netherlands is",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    if n <=",
    "My favourite food is",
    "The meaning of life is",
]
samp = []
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        lg = model(ids).logits[0, -1].numpy().astype(np.float64)
    order = np.argsort(-lg)
    top = order[:300]
    rest = lg[order[300:]]
    counts, edges = np.histogram(rest, bins=120)
    centers = (edges[:-1] + edges[1:]) / 2
    samp.append({"prompt": p,
                 "top": [[show(int(i)), round(float(lg[i]), 3)] for i in top],
                 "rest": [[round(float(c), 3), int(n)] for c, n in zip(centers, counts) if n > 0]})
write("sampling", "SAMPLING_DATA", samp)

# ---------- logit lens: top-5 per laag
lens_prompts = [
    ("The capital of the Netherlands is", " Amsterdam"),
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("One, two, three, four,", " five"),
    ("The opposite of hot is", " cold"),
]
lens = []
for p, target in lens_prompts:
    ids = tok(p, return_tensors="pt").input_ids
    tid = tok.encode(target)[0]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    layers = []
    for l, h in enumerate(hs):
        x = h[0, -1]
        if l < len(hs) - 1:
            x = model.transformer.ln_f(x)
        pr = torch.softmax(model.lm_head(x), -1)
        t5 = torch.topk(pr, 5)
        layers.append({"top": [[show(int(i)), round(float(v) * 100, 2)] for v, i in zip(t5.values, t5.indices)],
                       "target": round(float(pr[tid]) * 100, 2)})
    lens.append({"prompt": p, "target": target, "layers": layers})
write("logitlens", "LENS_DATA", lens)
