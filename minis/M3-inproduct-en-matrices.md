# 🟡 Mini M3: Het inproduct als "hoeveel lijken deze twee op elkaar?"

*Komt terug in: embeddings en unembedding (1.1), attention (1.2), eigenlijk overal*

Als je één wiskundig idee meeneemt uit deze hele module, laat het dit zijn. Bijna alles in een transformer komt neer op: **twee vectoren vermenigvuldigen en kijken hoe groot het getal is.**

## Het inproduct

Voor twee vectoren $\mathbf{a}$ en $\mathbf{b}$ van dezelfde lengte:

$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \dots$$

Dat is de rekenregel. Maar de **betekenis** zit in de meetkundige versie:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos(\varphi)$$

waarbij $\varphi$ de hoek tussen de twee vectoren is. Dus:

| Hoek | cos | Inproduct | Betekenis |
|------|-----|-----------|-----------|
| 0° (zelfde richting) | 1 | groot positief | "deze lijken op elkaar" |
| 90° (loodrecht) | 0 | nul | "hebben niets met elkaar te maken" |
| 180° (tegengesteld) | −1 | groot negatief | "zijn elkaars tegenpool" |

### Een voorbeeld om het te voelen

Stel dat we woorden beschrijven met drie "kenmerken": [is-een-dier, is-groot, is-gevaarlijk].

| Woord | vector |
|-------|--------|
| kat | [1,0 · 0,2 · 0,3] |
| tijger | [1,0 · 0,8 · 0,9] |
| muis | [1,0 · 0,0 · 0,0] |
| vrachtwagen | [0,0 · 1,0 · 0,6] |

- kat · tijger = 1·1 + 0,2·0,8 + 0,3·0,9 = **1,43**, ze lijken op elkaar
- kat · vrachtwagen = 0 + 0,2 + 0,18 = **0,38**, weinig gemeen
- tijger · vrachtwagen = 0 + 0,8 + 0,54 = **1,34**. Hé, best hoog: allebei groot en gevaarlijk!

Dat laatste is precies het punt: het inproduct meet gelijkenis **in de kenmerken die in de vector zitten**. In een echt model zijn er niet 3 maar 768 tot 16.000 kenmerken, en heeft niemand ze benoemd. Het model leert zelf welke richtingen handig zijn.

## Waar je dit terugziet

- **Unembedding (1.1):** "welk token komt hierna?" = "met welke token-embedding heeft mijn eindvector het grootste inproduct?"
- **Attention (1.2):** "naar welk token moet ik kijken?" = "welke key heeft het grootste inproduct met mijn query?"
- **Zoeken/RAG (module 3):** "welk document past bij deze vraag?" = "welke document-embedding heeft het grootste inproduct met de vraag-embedding?"

## Matrixvermenigvuldiging = heel veel inproducten tegelijk

Als je $C = AB$ uitrekent, is elk getal $C_{ij}$ het inproduct van **rij $i$ van $A$** met **kolom $j$ van $B$**. Meer is het niet.

Dus als $Q$ een matrix is met in elke rij de query van één token ($T \times d$), en $K$ net zo, dan is

$$QK^\top \quad (T \times d) \cdot (d \times T) = (T \times T)$$

een tabel met **álle** inproducten tussen alle queries en alle keys. Vakje $(i, j)$: "hoe goed past wat token $i$ zoekt bij wat token $j$ aanbiedt?" Dat is precies de attention-scoretabel uit figuur 1.6.

### De shape-regel (handig voor elke opdracht)

$$(a \times b) \cdot (b \times c) = (a \times c)$$

De binnenste dimensies moeten gelijk zijn en "verdwijnen", de buitenste blijven over. Als je code een shape-error geeft, schrijf dan de shapes op papier uit. Negen van de tien keer zie je het dan meteen.

### 🔬 Probeer het (2 minuten)

```python
import numpy as np
kat, tijger, muis, truck = map(np.array, ([1, .2, .3], [1, .8, .9], [1, 0, 0], [0, 1, .6]))
W = np.stack([kat, tijger, muis, truck])      # (4, 3): elke rij een woord
print(W @ W.T)                                 # (4, 4): alle inproducten tegelijk
```

Herken je de getallen uit de tabel hierboven op de juiste plekken?
