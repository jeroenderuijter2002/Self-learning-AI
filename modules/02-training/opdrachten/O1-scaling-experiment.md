# Opdracht O1: Meet je eigen scaling law

*Module 2* · ⏱️ 2–3 uur (waarvan ~30 min rekentijd) · 📖 eerst: [2.1 Pretraining](../notes/2.1-pretraining.md) en [2.2 Scaling laws](../notes/2.2-scaling-laws.md)

## Doel

De belangrijkste wetmatigheid van modern AI-onderzoek zelf meten: train vijf versies van je mini-GPT uit module 1, van 13 duizend tot 1,8 miljoen parameters, en laat zien dat de loss een machtswet volgt. Daarna leid je de Chinchilla-verdeling af en reken je uit wat een lab met een budget van GPT-3-formaat had moeten doen. Na deze opdracht kun je in een sollicitatiegesprek uitleggen waarom Llama 3 8B op 15 biljoen tokens is getraind.

## Deel A: de rekenregels (verplicht)

Open [`o1_scaling.py`](o1_scaling.py) en implementeer:

| Functie | Wat | Uit |
|---|---|---|
| `count_params(n_layer, n_embd)` | parameters van de GPT in [`minigpt.py`](minigpt.py), zonder embeddings | 1.3 en 2.2 |
| `training_flops(N, D)` | $C \approx 6ND$ | 2.1 |
| `fit_power_law(N, L)` | fit $L = E + A N^{-\alpha}$ zonder scipy | 2.2 |
| `chinchilla_loss`, `chinchilla_optimal` | de formule en de gesloten oplossing | 2.2 |

```bash
python o1_scaling.py
# Chinchilla-optimum bij C = 5.76e+23: N ≈ 72B, D ≈ 1.33T (18 tokens/parameter)
# Alle tests geslaagd!
```

De tests controleren je telling tegen de echte modellen (en tegen GPT-2: 85 miljoen parameters zonder embeddings), je FLOPs tegen het GPT-3-paper, je fit op data met bekende parameters, en je gesloten oplossing tegen een numerieke minimalisatie.

<details>
<summary>💡 Hint: parameters tellen</summary>

Per block: qkv is een $d \times 3d$-matrix plus $3d$ bias, de projectie $d \times d + d$, de MLP $d \times 4d + 4d$ en $4d \times d + d$, en twee LayerNorms van elk $2d$. Tel op: $12d^2 + 13d$. De "12" is $3 + 1$ (attention) $+ 4 + 4$ (MLP). Vergeet `ln_f` niet.
</details>

<details>
<summary>💡 Hint: de power-law-fit</summary>

Voor een vaste $E$ is $\log(L - E)$ lineair in $\log N$. Fit die lijn met `np.polyfit(np.log(N), np.log(L - E), 1)`; de helling is $-\alpha$, het snijpunt $\log A$. Doe dat voor 400 waarden van $E$ tussen 0 en `0.999 * L.min()`, en houd de $E$ met de kleinste kwadratische fout op $L$.
</details>

## Deel B: het experiment (verplicht)

```bash
python o1_scaling.py --train
```

Dit traint vijf modellen met [`minigpt.py`](minigpt.py) (dezelfde architectuur als in O3 van module 1, in één bestand) op tiny Shakespeare. Alle modellen krijgen precies hetzelfde: 2.000 stappen van 32 × 64 tokens, warmup plus cosine-schema. Op een laptop-CPU duurt het samen ongeveer 15–30 minuten; het grootste model kost het meest. Je krijgt `o1_scaling.png` met twee panelen, zoals figuur 2.3 in de notes.

Heb je je eigen O3-code van module 1 nog? Dan is het nog leerzamer om dat model te gebruiken. Het enige wat je nodig hebt is een functie die een model van een gegeven grootte traint en de validatie-loss teruggeeft.

**Beantwoord, met je eigen getallen:**
1. Wat is je $\alpha$? Vergelijk met de $\alpha_N \approx 0{,}076$ van Kaplan et al. Waarom zou die van jou zo anders zijn? (Denk aan: karakters versus tokens, een minuscule dataset, en kleine modellen.)
2. Met hoeveel daalt de loss als je het model 10 keer groter maakt? Klopt dat met $10^{-\alpha}$ toegepast op $L - E$?
3. Kijk naar het linkerpaneel. Bij welk compute-budget wordt het op één na grootste model ingehaald door het grootste? Wat zegt dat over de vraag "welk model moet ik bij budget $C$ trainen"?
4. Hoeveel tokens per parameter kreeg elk model? Welke modellen zijn volgens Chinchilla "ondergevoed" en welke "overgevoed"?

## Deel C: als consultant (aanbevolen)

Gebruik `chinchilla_optimal` met de Epoch-AI-constanten uit de tests:

1. **GPT-3 achteraf.** GPT-3 kostte ongeveer $3{,}14 \cdot 10^{23}$ FLOPs. Wat was volgens de fit de optimale verdeling? Hoeveel lager was de voorspelde loss dan die van de echte keuze (175B, 300B tokens)?
2. **Llama 3 8B.** Reken uit hoeveel FLOPs 8 miljard parameters op 15 biljoen tokens kosten. Welk model was bij dat budget compute-optimaal geweest? Hoeveel slechter is Llama 3 8B volgens de formule? En hoeveel goedkoper is het per gegenereerd token dan dat optimale model?
3. **Het break-evenpunt.** Stel: het compute-optimale model is $k$ keer groter dan het overgetrainde model, en beide halen dezelfde kwaliteit (dat klopt niet precies, maar als schatting). Na hoeveel gegenereerde tokens is de extra trainingscompute terugverdiend? Gebruik $2N$ FLOPs per gegenereerd token.

> ▶ **Ter controle:** de [Scaling-calculator](https://jeroenderuijter2002.github.io/Self-learning-AI/interactief/scaling.html) gebruikt dezelfde formule. Je antwoorden op deel C moeten overeenkomen met wat de speeltuin laat zien.

## Deel D: uitbreidingen (kies er één)

1. **Data in plaats van parameters.** Houd het model vast op (3, 96) en varieer het aantal stappen: 250, 500, 1000, 2000, 4000. Fit $L(D) = E + B/D^\beta$. Let op: na een paar epochs gaat tiny Shakespeare herhalen. Zie je dat in je curve?
2. **Een echte isoFLOP-curve.** Kies één budget, bijvoorbeeld $C = 2 \cdot 10^{13}$ FLOPs, en train vijf modellen van verschillende grootte die elk precies dat budget opmaken (pas `max_steps` aan zodat $6ND = C$). Plot de eindloss tegen $N$. Heb je een dal, zoals in figuur 2.4?
3. **Bits per teken.** Reken je losses om naar bits per teken (deel door $\ln 2$). Hoe verhoudt het beste model zich tot een gewone compressor? (Hint: `gzip -9` op `tinyshakespeare.txt`, en deel de bestandsgrootte in bits door het aantal tekens.)

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je een eigen figuur hebt met minstens vijf punten en een fit, en je $\alpha$ en $E$ kunt interpreteren;
- [ ] je zonder te kijken $N_{\text{opt}}(C)$ kunt afleiden uit $E + A/N^\alpha + B/D^\beta$ en $C = 6ND$.

**Oplossing:** [`../oplossingen/o1_scaling_oplossing.py`](../oplossingen/o1_scaling_oplossing.py)
