# Zelftoets: Module 1

Beantwoord eerst zelf (op papier of hardop), klik dan pas het antwoord open. Weet je een vraag niet? Ga dan terug naar de genoemde notes.

---

## A. Tokens en embeddings ([1.1](notes/1.1-tokens-en-embeddings.md))

**A1.** Waarom start BPE met bytes in plaats van met Unicode-karakters?
<details><summary>Antwoord</summary>
Er zijn maar 256 bytes, en élke tekst (elke taal, emoji, binaire rommel) is een rij bytes. Er bestaat dus nooit een "onbekend" token. Met Unicode-karakters (~150.000) zou je basisvocabulaire al enorm zijn en zou je zeldzame karakters missen.
</details>

**A2.** Je verdubbelt de vocabulaire van 50k naar 100k. Waar komen de extra parameters terecht, en wat zijn voor- en nadelen?
<details><summary>Antwoord</summary>
In de embedding-matrix $E$ ($V \times d$) en de unembedding $W_U$ (bij weight tying is dat dezelfde matrix). Voordeel: betere compressie, dus kortere sequenties en meer tekst per context window, en minder rekenwerk in attention. Nadeel: meer parameters, een duurdere softmax over meer klassen, en zeldzame tokens krijgen minder trainingssignaal.
</details>

**A3.** Waarom heeft "bank" na de embedding-laag dezelfde vector in "ik zit op de bank" en "ik werk bij de bank", en waar wordt dat opgelost?
<details><summary>Antwoord</summary>
De embedding is een lookup per token-id, zonder context. Attention-lagen mengen informatie van omringende tokens in de vector, zodat die na een paar lagen contextafhankelijk wordt.
</details>

## B. Attention ([1.2](notes/1.2-attention.md))

**B1.** Schrijf de attention-formule op en geef de shape van elk tussenresultaat voor $T$ tokens en dimensie $d_k$.
<details><summary>Antwoord</summary>

$\text{softmax}(QK^\top/\sqrt{d_k} + M)V$. $Q, K, V$: $(T, d_k)$. $QK^\top$: $(T, T)$. Na softmax: $(T, T)$, rijen tellen op tot 1. Output: $(T, d_k)$.
</details>

**B2.** Alle keys zijn identiek. Hoe zien de attention-gewichten eruit (met causal mask)?
<details><summary>Antwoord</summary>
Elke query geeft dan dezelfde score voor alle toegestane posities, dus uniform: token $i$ geeft gewicht $1/i$ aan elk van de tokens $1..i$. De output is het lopende gemiddelde van de values. Dit is precies Karpathy's "wiskundige truc" uit de video.
</details>

**B3.** Waarom is attention zonder positie-informatie een probleem voor taal?
<details><summary>Antwoord</summary>
Attention is permutatie-equivariant: het werkt op een verzameling, niet op een rij. "Hond bijt man" en "man bijt hond" zouden (op volgorde na) dezelfde representaties geven. Positie-informatie (geleerd, sinusoïdaal of RoPE) breekt die symmetrie.
</details>

**B4.** Wat is het doel van de causal mask tijdens **training**? (Niet alleen tijdens generatie!)
<details><summary>Antwoord</summary>
Zo kun je in één forward pass voor alle $T$ posities tegelijk het volgende token voorspellen: $T$ trainingsvoorbeelden per sequentie. Zonder mask zou positie $t$ het antwoord ($t+1$) kunnen zien en leert het model niets bruikbaars.
</details>

**B5.** Een collega zegt: "We verdubbelen de context van 64k naar 128k tokens, dat kost 2× zoveel rekenwerk." Klopt dat?
<details><summary>Antwoord</summary>
Niet voor het attention-deel: $QK^\top$ schaalt met $T^2$, dus ~4× zoveel. De MLP's schalen lineair (2×). Het totaal ligt ertussenin, afhankelijk van welk deel domineert. Bij zeer lange contexten domineert attention.
</details>

## C. Architectuur ([1.3](notes/1.3-transformer-block.md))

**C1.** Teken een pre-norm transformer-block en schrijf het in twee regels code.
<details><summary>Antwoord</summary>

`x = x + attn(ln1(x))` en `x = x + mlp(ln2(x))`.
</details>

**C2.** Wat gebeurt er als je alle MLP's weglaat?
<details><summary>Antwoord</summary>
Attention is (los van de softmax-gewichten) een lineaire combinatie van values. Zonder MLP's mist het model per-token niet-lineaire verwerking, met veel minder capaciteit om kennis op te slaan of complexe functies te leren. Ook verdwijnt ~2/3 van de parameters. Zo'n model presteert duidelijk slechter (test het in de ablation van O3!).
</details>

**C3.** Waarom zijn residual connections essentieel voor een model met 96 lagen?
<details><summary>Antwoord</summary>
1) Gradiënten stromen direct via de identiteitsverbinding terug, zonder te verdwijnen. 2) Elke laag hoeft alleen een aanpassing (delta) te leren in plaats van een volledige transformatie. 3) Het creëert de residual stream als gedeeld communicatiekanaal tussen lagen.
</details>

**C4.** Reken uit: hoeveel parameters heeft de attention van één laag GPT-3 ($d_\text{model} = 12.288$)?
<details><summary>Antwoord</summary>

$4 \times 12.288^2 \approx 604$ miljoen (Q, K, V, O, biases genegeerd). × 96 lagen ≈ 58 miljard parameters in attention alleen.
</details>

## D. Generatie en modern ([1.4](notes/1.4-van-model-naar-tekst.md))

**D1.** Temperatuur → 0. Wat gebeurt er wiskundig?
<details><summary>Antwoord</summary>

$\text{softmax}(z/\tau)$ met $\tau \to 0$ convergeert naar een one-hot op de argmax. Daarom implementeert men $\tau = 0$ als greedy decoding (delen door 0 kan niet).
</details>

**D2.** Waarom zijn output-tokens bij API's duurder dan input-tokens?
<details><summary>Antwoord</summary>
De input (prompt) wordt in één parallelle forward pass verwerkt (de "prefill"), wat GPU's efficiënt kunnen. Output moet sequentieel, één token per forward pass, en is vaak beperkt door geheugenbandbreedte (het hele model en de KV-cache inlezen per token). Per token is output dus veel minder efficiënt.
</details>

**D3.** Waarom werkt een KV-cache alleen dankzij de causal mask?
<details><summary>Antwoord</summary>
Door de mask hangen de keys en values van token $i$ alleen af van tokens $\le i$. Nieuwe tokens veranderen ze dus nooit, waardoor je ze veilig kunt hergebruiken. Bij BERT (geen mask) verandert elke representatie zodra er een token bijkomt.
</details>

**D4.** Wat is het voordeel van Mixture-of-Experts in één zin?
<details><summary>Antwoord</summary>
Je krijgt de capaciteit (kennis) van heel veel parameters, maar per token gebruik je er maar een paar experts van, dus het rekenwerk per token blijft laag.
</details>

## E. De eindvraag 🎓

**E1.** Leg in max. 5 zinnen uit aan een slimme niet-techneut (bijvoorbeeld je vader) hoe Claude een antwoord produceert. Gebruik minstens de woorden *token*, *attention* en *volgende woord*.

**E2.** Leg hetzelfde uit aan een ML-engineer bij een sollicitatie, in max. 8 zinnen, met de juiste technische termen.

*(Geen modelantwoord. Vergelijk het met de samenvatting onderaan [1.4](notes/1.4-van-model-naar-tekst.md), of laat Claude je antwoord kritisch beoordelen.)*

---

**Alles goed?** 🎉 Dan ben je klaar voor **Module 2: hoe een model als Claude getraind wordt.**
