# Opdracht O4: Test-time compute en verifieerbare beloningen

*Module 2* · ⏱️ 1,5–3 uur · 📖 eerst: [2.6 Reasoning-modellen](../notes/2.6-reasoning-modellen.md)

## Doel

De rekenkern achter reasoning-modellen zelf bouwen en meten: hoe haal je een antwoord uit een gedachtegang, hoe combineer je meerdere pogingen (majority vote), hoe meet je eerlijk hoeveel "meer pogingen" oplevert (pass@k), en hoe rekent GRPO daar een leersignaal van uit. Daarna analyseer je echte samples van een klein model, en kun je beargumenteren wanneer extra rekenwerk tijdens het antwoorden de moeite waard is.

## Deel A: de bouwstenen (verplicht)

Open [`o4_test_time.py`](o4_test_time.py) en implementeer:

| Functie | Wat |
|---|---|
| `extract_answer(text)` | het eindantwoord uit `\boxed{...}`, of anders het laatste getal |
| `majority_vote(answers)` | het vaakst voorkomende antwoord (self-consistency) |
| `pass_at_k(n, c, k)` | de zuivere schatter $1 - \binom{n-c}{k}/\binom{n}{k}$ |
| `grpo_advantages(rewards)` | $(r_i - \bar r)/\sigma_r$ binnen één groep |

```bash
python o4_test_time.py
# Alle tests geslaagd!
```

De test van `pass_at_k` vergelijkt je formule met brute kracht: alle deelverzamelingen van $k$ uit $n$ samples aflopen. Als die twee overeenkomen, heb je het begrepen.

<details>
<summary>💡 Hint: waarom niet gewoon 1 − (1 − c/n)^k?</summary>

Die formule hoort bij trekken **met** teruglegging: alsof elke poging onafhankelijk een kans $c/n$ heeft. Maar je hebt een vaste verzameling van $n$ samples, en een groep van $k$ samples bevat elk sample hooguit één keer. Voor $k$ dicht bij $n$ maakt dat veel uit: met $n = k$ moet pass@k precies 1 zijn als $c \geq 1$.
</details>

## Deel B: echte samples analyseren (verplicht)

```bash
python o4_test_time.py --analyse
```

Dit leest de metingen achter figuur 2.11 ([`tools/results/m2_gsm8k.json`](../../../tools/results/m2_gsm8k.json)): voor elke GSM8K-vraag het juiste antwoord, het greedy antwoord en 16 gesamplede antwoorden van Qwen2.5-0.5B-Instruct.

**Beantwoord:**
1. Hoeveel verbetert majority vote ten opzichte van één greedy antwoord bij $k = 16$? En pass@16? Wat is het verschil tussen die twee getallen waard, als je een perfecte verifier had?
2. Voor hoeveel vragen geven alle 16 samples hetzelfde (goede of foute) antwoord? Wat betekent dat voor een GRPO-training met deze vragen?
3. Reken uit wat 16 samples kosten in tokens vergeleken met één greedy antwoord (de gemiddelde lengte staat in het veld `len`). Bij welke prijs per miljoen output-tokens is majority vote op 1.000 vragen per dag duurder dan een model dat twee keer zo groot is? Maak een aanname over hoe goed dat grotere model is, en zeg welke.

## Deel C: zelf samplen (optioneel, Colab-GPU aangeraden)

```bash
python o4_test_time.py --genereer --n 20 --k 8
```

Op een CPU kost dit ongeveer een minuut per vraag; op een gratis Colab-T4 een paar seconden.

1. **Prompt-experiment.** Verander de prompt in `genereer`: haal *"Please reason step by step"* weg en vraag om alleen het antwoord. Hoeveel daalt de greedy-score? Dit is chain-of-thought in één experiment.
2. **Temperatuur.** Doe majority vote met temperatuur 0,3, 0,7 en 1,2. Bij welke temperatuur helpt stemmen het meest, en waarom niet bij de laagste?
3. **Een groter model.** Herhaal met Qwen2.5-1.5B-Instruct. Wat is beter: het kleine model met 8 samples, of het grote met 1? Zet de kosten (aantal parameters × aantal tokens) ernaast.

## Deel D: een mini-GRPO (uitdaging)

Je hebt alles om de kern van GRPO te schrijven. Neem je mini-GPT uit O1 (of een klein model) en een speelgoedtaak met een verifieerbare beloning, bijvoorbeeld: *"genereer een rij van 8 tekens die eindigt met een klinker"*, of optellen van twee cijfers. Per stap: sample 8 antwoorden per prompt, bereken `grpo_advantages`, en maximaliseer $\sum_i A_i \log \pi_\theta(y_i)$ (gebruik `sequence_logprob` uit O2). Zie je de beloning stijgen? Wat gebeurt er met de diversiteit van de antwoorden?

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je met de echte data kunt laten zien wat majority vote en een verifier opleveren, en wat het kost;
- [ ] je kunt uitleggen waarom een vraag waarop alle samples goed zijn, bij GRPO geen leersignaal geeft.

**Oplossing:** [`../oplossingen/o4_test_time_oplossing.py`](../oplossingen/o4_test_time_oplossing.py)
