# 🟡 Mini M1: Softmax en temperatuur

*Komt terug in: attention (1.2), sampling (1.4), en later RLHF en distillation*

## Softmax

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Zet willekeurige reële getallen (logits) om naar een kansverdeling. Drie eigenschappen die je moet kennen:

1. **Verschuivingsinvariant:** $\text{softmax}(\mathbf{z} + c) = \text{softmax}(\mathbf{z})$. Daarom mag je voor numerieke stabiliteit $\max(\mathbf{z})$ aftrekken (O2!), zodat $e^{1000}$ nooit overflowt.
2. **Niet schaalinvariant:** $\text{softmax}(2\mathbf{z}) \neq \text{softmax}(\mathbf{z})$. Grotere logits geven een scherpere verdeling. Dat is de basis van temperatuur en van de $\sqrt{d_k}$-schaling.
3. **"Zachte argmax":** het is een differentieerbare benadering van "kies de grootste". Daarom heet het soft-max.

## Temperatuur

$$p_i = \text{softmax}(\mathbf{z} / \tau)_i$$

| $\tau$ | Effect | Voorbeeld met logits [2, 1, 0] |
|--------|--------|--------------------------------|
| 0,1 | bijna one-hot | [1,00, 0,00, 0,00] |
| 0,5 | scherp | [0,87, 0,12, 0,02] |
| 1,0 | origineel | [0,67, 0,24, 0,09] |
| 2,0 | vlakker | [0,51, 0,31, 0,19] |
| ∞ | uniform | [0,33, 0,33, 0,33] |

**De naam komt uit de statistische fysica:** de Boltzmann-verdeling $p_i \propto e^{-E_i/kT}$. Bij hoge temperatuur zijn alle toestanden even waarschijnlijk (chaos), bij lage temperatuur zit alles in de laagste-energietoestand.

### 🔬 Probeer het (5 minuten)

```python
import numpy as np
z = np.array([2.0, 1.0, 0.0])
for tau in [0.1, 0.5, 1, 2, 10]:
    p = np.exp(z / tau) / np.exp(z / tau).sum()
    entropy = -(p * np.log(p + 1e-12)).sum()
    print(f"tau={tau:4}: p={np.round(p, 2)}, entropie={entropy:.2f}")
```

Wat is de maximale entropie voor 3 klassen? ($\ln 3 \approx 1{,}10$.) Bij welke $\tau$ kom je in de buurt?

## Waar zie je dit terug?
- **Attention:** $QK^\top / \sqrt{d_k}$ is eigenlijk temperatuur $\tau = \sqrt{d_k}$ → [Mini M2](M2-waarom-wortel-dk.md)
- **Claude API:** de `temperature`-parameter is letterlijk deze $\tau$
- **Knowledge distillation (module 4):** een groot model traint een klein model met "zachte labels" bij hoge $\tau$
