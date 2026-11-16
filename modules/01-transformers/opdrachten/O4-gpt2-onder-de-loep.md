# Opdracht O4: GPT-2 onder de loep

*Module 1* · ⏱️ 1–2 uur · 📖 eerst: [1.4 Van model naar tekst](../notes/1.4-van-model-naar-tekst.md)

## Doel
Je hebt een speelgoedmodel gebouwd. Nu open je een **echt** voorgetraind model (GPT-2, 124M parameters) en controleer je alles wat je geleerd hebt: parameters, attention, sampling en de KV-cache.

## Setup

```bash
pip install transformers torch matplotlib
```

```python
import torch, time
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
# "eager" = de gewone attention-implementatie; nodig om attention-gewichten terug te krijgen in deel 3
```

---

## Deel 1: parameters tellen 🔢

```python
print(sum(p.numel() for p in model.parameters()))
for name, p in list(model.named_parameters())[:16]:
    print(f"{name:45s} {tuple(p.shape)}")
```

**Vragen:**
- Komt het totaal overeen met je berekening uit [1.3](../notes/1.3-transformer-block.md)?
- GPT-2 slaat Q, K en V op in **één** matrix `c_attn` met shape `(768, 2304)`. Waarom 2304?
- Check de weight tying: is `model.lm_head.weight` hetzelfde object als `model.transformer.wte.weight`?

## Deel 2: next-token prediction zien 🎯

```python
prompt = "The capital of the Netherlands is"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1]
probs = torch.softmax(logits, -1)
top = torch.topk(probs, 10)
for p, i in zip(top.values, top.indices):
    print(f"{tok.decode(i)!r:15s} {p:.3f}")
```

**Vragen:**
- Staat " Amsterdam" bovenaan? Met welke kans?
- Probeer prompts waar het model moet "redeneren", zoals `"If I have 3 apples and eat one, I have"`. Waar gaat een 124M-model de mist in?

## Deel 3: attention patterns 👀

```python
import matplotlib.pyplot as plt
text = "When Mary and John went to the store, John gave a drink to"
ids = tok(text, return_tensors="pt").input_ids
out = model(ids, output_attentions=True)
tokens = [tok.decode(t) for t in ids[0]]

layer, head = 9, 9                       # probeer verschillende!
A = out.attentions[layer][0, head].detach()
plt.figure(figsize=(7, 6))
plt.imshow(A, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title(f"Laag {layer}, head {head}"); plt.tight_layout(); plt.show()
```

**Opdracht:** zoek door verschillende lagen en heads en vind (tip: in de [Attention-speeltuin](https://jeroenderuijter2002.github.io/Self-learning-AI/interactief/attention.html) zie je alle 144 heads tegelijk, handig om eerst te verkennen):
- een **previous-token head**: een diagonaal net onder de hoofddiagonaal;
- een head die sterk naar het **eerste token** kijkt (een "attention sink", als een soort "niks te doen"-stand);
- een head die vanaf de laatste positie naar **" Mary"** kijkt. Het model moet " Mary" voorspellen, niet " John"!

> Dit zinnetje komt uit beroemd interpretability-onderzoek naar "indirect object identification". Er bestaan echt circuits van heads die dit oplossen. Meer daarover in module 7.

## Deel 4: temperatuur en sampling 🌡️

```python
prompt = tok("Once upon a time in Eindhoven,", return_tensors="pt").input_ids
for T in [0.3, 0.7, 1.0, 1.5]:
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=40, do_sample=True,
                         temperature=T, top_k=0, pad_token_id=tok.eos_token_id)
    print(f"\n--- T = {T} ---\n{tok.decode(out[0])}")
```

**Vragen:** beschrijf het verschil. Probeer daarna `top_p=0.9` bij $T = 1{,}5$. Wordt de chaos getemd? Waarom?

## Deel 5: de KV-cache meten ⚡

```python
prompt = tok("The history of artificial intelligence", return_tensors="pt").input_ids
for use_cache in [False, True]:
    t0 = time.time()
    model.generate(prompt, max_new_tokens=200, do_sample=False,
                   use_cache=use_cache, pad_token_id=tok.eos_token_id)
    print(f"use_cache={use_cache}: {time.time() - t0:.2f} s")
```

**Vragen:**
- Hoeveel sneller is het met cache?
- Herhaal met `max_new_tokens` = 50, 100, 200 en 400. Groeit het verschil? Verklaar dit met de complexiteit uit [1.4](../notes/1.4-van-model-naar-tekst.md).
- **Rekenopdracht:** hoeveel geheugen kost de KV-cache van GPT-2 voor 1024 tokens in float32? Formule: $2 \times n_\text{layers} \times T \times d_\text{model} \times 4$ bytes. Doe hetzelfde voor een hypothetisch model met 80 lagen, $d_\text{model}$ = 8192 en 100.000 tokens context in bfloat16 (2 bytes). Wat zegt dit over waarom lange context duur is?

---

## ✅ Je bent klaar als
- [ ] je de drie soorten heads uit deel 3 gevonden hebt (screenshots bewaren!);
- [ ] je met eigen metingen kunt onderbouwen waarom de KV-cache bestaat.
