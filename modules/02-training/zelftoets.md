# Zelftoets: Module 2

Beantwoord eerst zelf (op papier of hardop), klik dan pas het antwoord open. Weet je een vraag niet? Ga dan terug naar de genoemde notes.

---

## A. Pretraining ([2.1](notes/2.1-pretraining.md))

**A1.** Een basismodel krijgt de prompt *"Wat is de hoofdstad van Frankrijk?"* en antwoordt met *"Wat is de hoofdstad van Duitsland? Wat is de hoofdstad van Italië?"*. Leg uit waarom dit precies het gedrag is waarvoor het getraind is.
<details><summary>Antwoord</summary>
Pretraining maximaliseert de kans op het volgende token in webdocumenten. Een regel met een vraag staat op internet vaak in een lijst met meer vragen (een quiz, een FAQ, een overhoring). Het model maakt het meest waarschijnlijke document af; het heeft geen begrip van "een gebruiker die een antwoord wil". Dat leert het pas in SFT.
</details>

**A2.** Schat het rekenwerk van het trainen van een model met 7 miljard parameters op 2 biljoen tokens. Hoeveel GPU-dagen is dat bij een effectieve snelheid van $4 \cdot 10^{14}$ FLOP/s?
<details><summary>Antwoord</summary>

$C \approx 6ND = 6 \cdot 7 \cdot 10^9 \cdot 2 \cdot 10^{12} = 8{,}4 \cdot 10^{22}$ FLOPs. Gedeeld door $4 \cdot 10^{14}$ is $2{,}1 \cdot 10^8$ seconden, ofwel ongeveer 2.400 GPU-dagen: een maand op zo'n 80 GPU's.
</details>

**A3.** Waarom gebruik je een warmup in het learning-rate-schema? Noem twee redenen.
<details><summary>Antwoord</summary>
(1) Aan het begin zijn de gradiënten groot en chaotisch (willekeurige gewichten), en een grote stap kan het model in een slecht gebied gooien waar het niet meer uitkomt. (2) Adam deelt door een lopend gemiddelde van de gekwadrateerde gradiënten; in de eerste stappen is die schatting nog onbetrouwbaar, waardoor de effectieve stapgrootte onvoorspelbaar is. Een warmup geeft de optimizer tijd om die statistiek op te bouwen.
</details>

**A4.** Waarom is deduplicatie van trainingsdata belangrijk? Geef twee redenen.
<details><summary>Antwoord</summary>
Dubbele tekst kost rekenwerk zonder iets nieuws te leren, en het model gaat herhaalde passages letterlijk onthouden en opdreunen (Lee et al. vonden tien keer minder letterlijk geciteerde tekst na deduplicatie). Bonus: duplicaten tussen train- en testset maken je evaluatie te optimistisch.
</details>

## B. Scaling laws ([2.2](notes/2.2-scaling-laws.md))

**B1.** Wat betekent het dat de loss een "rechte lijn op log-log-schaal" volgt, en waarom is dat zo nuttig voor een lab?
<details><summary>Antwoord</summary>

Dat $L \approx c \cdot N^{-\alpha}$ (een machtswet), eventueel plus een constante $E$. Het nut: je traint een reeks kleine, goedkope modellen, fit de lijn, en voorspelt de loss van een honderd of duizend keer groter model voordat je het traint. OpenAI deed dat voor GPT-4.
</details>

**B2.** In de Chinchilla-formule $L = E + A/N^\alpha + B/D^\beta$: wat betekent elk van de drie termen?
<details><summary>Antwoord</summary>

$E$: de onvermijdelijke loss, omdat tekst niet volledig voorspelbaar is (de entropie van taal). $A/N^\alpha$: de straf voor een te klein model; verdwijnt als $N \to \infty$. $B/D^\beta$: de straf voor te weinig data; verdwijnt als $D \to \infty$.
</details>

**B3.** Je budget wordt 100 keer groter. Hoeveel groter maak je volgens Chinchilla het model, en hoeveel meer data gebruik je?
<details><summary>Antwoord</summary>

Met $\alpha \approx \beta$ groeien beide als $\sqrt{C}$: het model ongeveer 10 keer groter, de data ongeveer 10 keer meer. ($10 \times 10 = 100$.)
</details>

**B4.** Llama 3 8B zag 15 biljoen tokens, bijna 1.900 per parameter, ver boven de Chinchilla-20. Was dat een fout?
<details><summary>Antwoord</summary>
Nee. Chinchilla minimaliseert de loss voor een vast <i>trainings</i>budget. Een model dat veel gebruikt wordt, kost ook rekenwerk bij elke aanroep (~2N FLOPs per token). Een kleiner model dat langer is getraind is iets slechter dan het optimum voor dat trainingsbudget, maar veel goedkoper in gebruik, en dat verdient de extra training terug. Dit heet inference-aware scaling (Sardana et al., 2024).
</details>

**B5.** Een collega toont een grafiek waarin een vaardigheid "plotseling verschijnt" boven een bepaalde modelgrootte. Welke vraag stel je eerst?
<details><summary>Antwoord</summary>
"Met welke maatstaf is dat gemeten?" Exact match en andere alles-of-niets-maatstaven maken van een geleidelijke verbetering een sprong: als elk token met kans $p$ goed is, is een antwoord van $k$ tokens met kans $p^k$ goed. Vraag om een gladde maatstaf (log-kans, per-token-nauwkeurigheid) op dezelfde uitvoer (Schaeffer et al., 2023).
</details>

## C. Supervised fine-tuning ([2.3](notes/2.3-supervised-fine-tuning.md))

**C1.** Wat is het verschil tussen de loss in pretraining en die in SFT?
<details><summary>Antwoord</summary>
Dezelfde volgende-token-cross-entropy. Twee verschillen: de data (gesprekken in een chat-template in plaats van webdocumenten) en het loss-masker (alleen de tokens van het antwoord van de assistent tellen mee, niet de vraag van de gebruiker of de systeemprompt).
</details>

**C2.** Waarom moet het `<|im_end|>`-token aan het eind van een antwoord wél in de loss meetellen?
<details><summary>Antwoord</summary>
Zo leert het model wanneer het antwoord klaar is. Zonder dat signaal blijft het doorgaan, vaak met een verzonnen vervolgvraag van de "gebruiker", net als een basismodel.
</details>

**C3.** Wat is de Superficial Alignment Hypothesis, en welke meting uit deze module ondersteunt haar?
<details><summary>Antwoord</summary>
De hypothese (LIMA, 2023) dat bijna alle kennis en vaardigheden uit pretraining komen, en dat alignment het model vooral leert welke stijl en welk format het moet gebruiken. Steun: bij de token-rang-meting (figuur 2.8, Lin et al. 2023) kiest het basismodel op de meeste posities van een assistent-antwoord exact hetzelfde token; de verschillen zitten vooral in stijltokens aan het begin en in opmaak.
</details>

**C4.** Leg uit waarom LoRA met rang 16 op een $4096 \times 4096$-matrix 128 keer minder parameters traint. Waarom begint $B$ op nul?
<details><summary>Antwoord</summary>

LoRA traint $A$ ($16 \times 4096$) en $B$ ($4096 \times 16$): $2 \cdot 16 \cdot 4096 = 131.072$ parameters, tegen $4096^2 = 16.777.216$; de verhouding is $4096 / (2 \cdot 16) = 128$. $B = 0$ zorgt dat $BA = 0$, zodat het model bij de start precies het basismodel is; $A$ is willekeurig zodat $B$ meteen een gradiënt krijgt.
</details>

**C5.** Een klant wil een model finetunen op zijn productcatalogus "zodat het alle producten kent". Wat adviseer je, en waarom?
<details><summary>Antwoord</summary>
Gebruik RAG (module 3) voor de kennis en fine-tuning hooguit voor toon en format. Fine-tuning voegt nieuwe feiten maar langzaam en onbetrouwbaar toe, en training op antwoorden met feiten die het model niet kent, kan het zelfs leren te hallucineren (Gekhman et al., 2024). Bovendien verandert een catalogus, en een model hertrain je niet elke week.
</details>

## D. RLHF ([2.4](notes/2.4-rlhf.md))

**D1.** Schrijf het Bradley-Terry-model en de loss van een reward model op. Waarom heeft een reward-score van 3,7 op zichzelf geen betekenis?
<details><summary>Antwoord</summary>

$P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$, loss $= -\log \sigma(r_w - r_l)$. Alleen verschillen tussen scores komen in het model voor; je kunt bij alle scores een constante optellen zonder dat er iets verandert.
</details>

**D2.** Zet de RL-stap van RLHF in MDP-termen: wat zijn toestand, actie, overgang en beloning?
<details><summary>Antwoord</summary>
Toestand: de prompt plus de tot nu toe gegenereerde tokens. Actie: het volgende token. Overgang: deterministisch, plak het token erachter. Beloning: nul bij elk token, en aan het eind de score van het reward model, min een KL-straf ten opzichte van het referentiemodel (in de praktijk vaak per token verrekend).
</details>

**D3.** Waarom zit er een KL-straf in het RLHF-doel? Wat gebeurt er met $\beta = 0$?
<details><summary>Antwoord</summary>

Het reward model is alleen betrouwbaar voor antwoorden die lijken op waar het op getraind is (die van het SFT-model). Zonder rem ($\beta = 0$) zoekt de policy de plekken waar het reward model fouten maakt: reward hacking, rare of extreem lange antwoorden, en verlies van taalvaardigheid. De KL-term werkt als een zachte trust region, net als een straf op afwijking in MPC.
</details>

**D4.** Noem drie vormen van reward hacking of overoptimalisatie die in de notes genoemd worden.
<details><summary>Antwoord</summary>
(1) Overoptimalisatie (Gao et al.): de score van het proxy-reward model blijft stijgen terwijl de "echte" kwaliteit na een punt daalt. (2) Lengte: reward models belonen langere antwoorden, en een PPO-run met alleen lengte als beloning haalt al veel van de RLHF-winst (Singhal et al.). (3) Sycophancy: het model zegt wat de gebruiker wil horen, omdat mensen (en dus reward models) dat waarderen (Sharma et al.).
</details>

## E. Constitutional AI en DPO ([2.5](notes/2.5-constitutional-ai-en-dpo.md))

**E1.** Beschrijf de twee fasen van Constitutional AI.
<details><summary>Antwoord</summary>
(1) Supervised: een helpful-only model beantwoordt schadelijke prompts, bekritiseert zijn eigen antwoord volgens een principe uit de constitutie, en herschrijft het; het model wordt gefinetuned op de herschreven antwoorden. (2) RLAIF: een model beoordeelt welke van twee antwoorden beter past bij een principe; die AI-voorkeuren trainen het reward model, waarna gewone RL volgt.
</details>

**E2.** Leid in vijf regels de DPO-loss af. Waar verdwijnt de partitiefunctie $Z(x)$?
<details><summary>Antwoord</summary>

1. Het doel $\max_\pi \mathbb{E}[r] - \beta \mathrm{KL}(\pi \Vert \pi_{\text{ref}})$ heeft als oplossing $\pi^* = \pi_{\text{ref}} e^{r/\beta} / Z(x)$.
2. Omdraaien: $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z(x)$.
3. Bradley-Terry hangt alleen af van $r(x, y_w) - r(x, y_l)$.
4. In dat verschil valt $\beta \log Z(x)$ weg, want het hangt niet van $y$ af.
5. Vervang $\pi^*$ door $\pi_\theta$: $\mathcal{L} = -\log\sigma\big(\beta \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$.
</details>

**E3.** Wat is de waarde van de DPO-loss bij de start van de training, en waarom?
<details><summary>Antwoord</summary>

$\ln 2 \approx 0{,}693$. Bij de start is $\pi_\theta = \pi_{\text{ref}}$, dus beide log-verhoudingen zijn 0, en $-\log\sigma(0) = \ln 2$.
</details>

**E4.** Noem een voordeel en een nadeel van DPO ten opzichte van PPO.
<details><summary>Antwoord</summary>
Voordeel: geen reward model, geen sampling tijdens de training, twee in plaats van vier modellen, stabiel en goedkoop (Llama 3 koos het om die reden). Nadeel: offline; het model leert alleen van vaste paren en verkent zelf niet, en kan onvoorspelbaar gedrag krijgen op antwoorden ver van de data. Goed afgestelde PPO wint op sommige taken (Xu et al., 2024).
</details>

## F. Reasoning-modellen ([2.6](notes/2.6-reasoning-modellen.md))

**F1.** Waarom is "meer tokens" letterlijk "meer nadenken" voor een transformer?
<details><summary>Antwoord</summary>
Per token doet het model een vaste hoeveelheid rekenwerk (één keer door alle lagen). Een antwoord dat in één token moet, krijgt dus maar één ronde. Een gedachtegang geeft elke tussenstap een eigen ronde, en via attention kan elke volgende stap alles teruglezen: de tekst is kladpapier.
</details>

**F2.** Wat is het verschil tussen pass@k en majority vote, en wat zegt het gat tussen die twee?
<details><summary>Antwoord</summary>
pass@k: de kans dat minstens één van $k$ samples goed is (je hebt een verifier nodig om dat te benutten). Majority vote: je kiest het vaakste antwoord, zonder verifier. Het gat is de waarde van een goede verifier: zo vaak zit het goede antwoord ertussen maar wint het niet bij de stemming.
</details>

**F3.** Wat is RLVR, en waarom heeft het minder last van reward hacking dan RLHF?
<details><summary>Antwoord</summary>
RL with Verifiable Rewards: RL op taken waarvan je het antwoord automatisch kunt controleren (wiskunde, code met tests). De beloning is een regel ("klopt het antwoord?") in plaats van een geleerd reward model, dus er is geen benadering om te bespelen. (Helemaal immuun is het niet: een model kan bijvoorbeeld tests omzeilen als die zwak zijn.)
</details>

**F4.** Hoe berekent GRPO het voordeel van een antwoord, en waarom is er geen critic nodig?
<details><summary>Antwoord</summary>

Voor elke vraag genereer je een groep van $G$ antwoorden, en $A_i = (r_i - \bar r)/\sigma_r$ binnen die groep. Het groepsgemiddelde is de baseline die bij PPO door de critic (een apart waardenetwerk) wordt geschat.
</details>

**F5.** Je leest de gedachtegang van een reasoning-model en ziet een nette redenering. Kun je erop vertrouwen dat het model zo tot zijn antwoord kwam?
<details><summary>Antwoord</summary>
Niet zonder meer. In het onderzoek van Anthropic (2025) noemden Claude 3.7 Sonnet en DeepSeek-R1 een hint die ze aantoonbaar gebruikten maar in 25% respectievelijk 39% van de gevallen. Een gedachtegang is nuttige informatie, geen betrouwbaar logboek.
</details>

---

## G. Het grote plaatje

**G1.** Leg in vijf zinnen uit, aan een collega zonder AI-achtergrond, waarom Claude een behulpzame assistent is en GPT-2 een tekstvoorspeller, terwijl de architectuur grotendeels dezelfde is.
<details><summary>Antwoord (een mogelijke versie)</summary>
Beide zijn transformers die het volgende woord voorspellen. Het verschil zit in de opleiding. Eerst leest het model biljoenen woorden tekst en leert het zo hoe de wereld en taal in elkaar zitten; daar stopte GPT-2. Daarna leert het met voorbeeldgesprekken hoe een behulpzame assistent antwoordt, en met oordelen van mensen en van een AI met geschreven principes welke antwoorden beter zijn. Tot slot oefent het op controleerbare taken, waardoor het leert eerst na te denken en zichzelf te controleren voordat het antwoordt.
</details>

---
[← Terug naar module 2](README.md)
