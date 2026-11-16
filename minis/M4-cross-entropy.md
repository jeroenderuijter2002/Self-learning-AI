# 🟡 Mini M4: Cross-entropy, of "hoe verrast was het model?"

*Komt terug in: notes 1.1 (de loss), opdracht O3 (de getallen die je ziet tijdens trainen), module 2 (pretraining)*

## Je kent het al

Weet je nog, uit elke classificatie-cursus: je hebt een model dat een kansverdeling over klassen voorspelt (softmax), en je traint het met **cross-entropy loss**:

$$\mathcal{L} = -\log p_\theta(\text{juiste klasse})$$

Een taalmodel is precies dit, met twee verschillen:
1. de "klassen" zijn de ~50.000 tokens in de vocabulaire;
2. je doet het voor **elke positie** in de tekst en neemt het gemiddelde.

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)$$

## De intuïtie: verrassing

Lees $-\log p$ als **"hoe verrast ben ik?"**:

| Kans die het model gaf aan het echte volgende token | $-\log p$ | Gevoel |
|---|---|---|
| 0,99 | 0,01 | "Zag ik aankomen" |
| 0,5 | 0,69 | "Had ik half verwacht" |
| 0,01 | 4,6 | "Huh?!" |
| 0,0001 | 9,2 | "Totaal onverwacht" |

Trainen betekent: **het model minder verrast maken door echte tekst**. Daarvoor moet het grammatica, feiten, stijl en redeneringen oppikken. Dat alles volgt uit deze ene, simpele doelstelling.

## Een getal om te onthouden: ln(V)

Een **onbetraind** model weet niets en geeft elk token ongeveer dezelfde kans, $1/V$. De loss is dan

$$-\log \frac{1}{V} = \ln V$$

| Vocabulaire | Loss bij start |
|---|---|
| 65 tekens (Shakespeare, O3) | ln 65 ≈ **4,17** |
| 50.257 tokens (GPT-2) | ln 50.257 ≈ **10,8** |

Dit is een geweldige sanity check. Begint je model in O3 met een loss van ~4,2, dan is de initialisatie goed. Begint het op 8 of 20, dan klopt er iets niet (bijvoorbeeld een veel te grote init, zodat het model in het begin al "zeker" is van onzin). Karpathy gebruikt deze truc ook in zijn video.

## Perplexity

Je ziet ook vaak **perplexity** = $e^{\mathcal{L}}$. Intuïtie: "het model twijfelt alsof het tussen zoveel opties moet kiezen." Een loss van 1,5 geeft perplexity 4,5: gemiddeld twijfelt het model tussen zo'n 4–5 tokens.

## 🔬 Probeer het

```python
import numpy as np
print(np.log(65), np.log(50257))      # verwachte start-loss
for loss in [4.17, 2.5, 1.5]:
    print(f"loss {loss} → perplexity {np.exp(loss):.1f}")
```

In O3 zie je de loss dalen van ~4,2 naar ~1,5. Vertaal dat nu zelf: van "twijfelt tussen alle 65 tekens" naar "twijfelt tussen ~4–5 tekens".
