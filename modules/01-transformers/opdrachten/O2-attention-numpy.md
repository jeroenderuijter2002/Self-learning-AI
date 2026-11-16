# Opdracht O2: Attention from scratch in NumPy

*Module 1* · ⏱️ 1–2 uur · 📖 eerst: [1.2 Attention](../notes/1.2-attention.md)

## Doel
De attention-formule omzetten in werkende code, zonder PyTorch. Als je dit kunt, kun je in een sollicitatiegesprek bij QuantumBlack of BCG X attention op een whiteboard uitschrijven.

> ▶ **Eerst een gevoel krijgen?** In de [Attention-speeltuin](https://jeroenderuijter2002.github.io/Self-learning-AI/interactief/attention.html) staat het rekenvoorbeeld uit figuur 1.6 met aanpasbare getallen. Je code moet precies dezelfde gewichten geven.

## Deel A: implementeren (verplicht)

Open [`o2_attention.py`](o2_attention.py) en implementeer:

| Functie | Wat |
|---------|-----|
| `softmax(x, axis)` | numeriek stabiel |
| `causal_mask(T)` | onderdriehoeksmatrix van `True` |
| `scaled_dot_product_attention(Q, K, V, mask)` | de kernformule |
| `multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)` | splitsen, attention, samenvoegen |

```bash
python o2_attention.py
# Max gewicht zonder schaling: 0.9xx | met schaling: 0.4xx
# Alle tests geslaagd!
```

De tests controleren onder meer:
- of elke rij attention-gewichten optelt tot 1;
- of er **geen toekomstlek** is: de bovendriehoek moet 0 zijn;
- of het veranderen van het laatste token geen invloed heeft op eerdere outputs. Dat is de echte causaliteitstest.

<details>
<summary>💡 Hint: de mask toepassen</summary>

Zet gemaskeerde scores op `-np.inf` **vóór** de softmax: `np.where(mask, scores, -np.inf)`. Na de softmax wordt $e^{-\infty} = 0$.
</details>

<details>
<summary>💡 Hint: heads splitsen</summary>

`(T, d_model)` → `reshape(T, n_heads, d_k)` → `transpose(1, 0, 2)` geeft `(n_heads, T, d_k)`. Als je `scaled_dot_product_attention` met `@` en `swapaxes(-1, -2)` schrijft, werkt die automatisch op alle heads tegelijk dankzij broadcasting.
</details>

## Deel B: experimenten (aanbevolen)

1. **De √d_k-schaling.** De test print het gemiddelde maximale gewicht met en zonder schaling. Waarom wordt de softmax zonder schaling bijna one-hot? Wat betekent dat voor de gradiënten? Lees [Mini M2](../../../minis/M2-waarom-wortel-dk.md) en check je verklaring.
2. **Visualiseer een attention pattern.**
   ```python
   import matplotlib.pyplot as plt
   _, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
   plt.imshow(w, cmap="viridis"); plt.colorbar(); plt.show()
   ```
3. **Ontwerp zelf een head.** Kun je $W_Q$ en $W_K$ met de hand kiezen zodat elk token (bijna) alleen naar het **vorige** token kijkt? Gebruik als input one-hot positievectoren, $X = I_T$. Dit heet een *previous-token head* en bestaat echt in getrainde modellen.

## Deel C: uitbreiding (optioneel)

Schrijf dezelfde `multi_head_attention` in **PyTorch** en controleer met `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` dat je dezelfde output krijgt. Dit is een opstapje naar O3.

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je zonder te kijken de shapes van $Q$, $K^\top$, $QK^\top$ en de output kunt opschrijven.

**Oplossing:** [`../oplossingen/o2_attention_oplossing.py`](../oplossingen/o2_attention_oplossing.py)
