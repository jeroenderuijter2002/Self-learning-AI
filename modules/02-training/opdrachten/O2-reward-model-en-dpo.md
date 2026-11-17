# Opdracht O2: Een reward model en de DPO-loss

*Module 2* · ⏱️ 2–3 uur · 📖 eerst: [2.4 RLHF](../notes/2.4-rlhf.md) en [2.5 Constitutional AI en DPO](../notes/2.5-constitutional-ai-en-dpo.md)

## Doel

De twee rekenkernen van alle preference-training zelf bouwen: het **Bradley-Terry-verlies** waarmee een reward model wordt getraind, en de **DPO-loss** die het reward model en de RL-stap in één klap vervangt. Daarna train je een echt (klein) reward model op de voorkeursdata die Anthropic publiceerde, en ontdek je zelf de lengte-bias uit de notes.

## Deel A: de bouwstenen (verplicht)

Open [`o2_preferences.py`](o2_preferences.py) en implementeer:

| Functie | Wat | Uit |
|---|---|---|
| `bradley_terry_prob(r_w, r_l)` | $\sigma(r_w - r_l)$ | 2.4 |
| `reward_model_loss(r_w, r_l)` | $-\log\sigma(r_w - r_l)$, numeriek stabiel | 2.4 |
| `sequence_logprob(logits, ids, mask)` | $\log \pi(y \mid x)$: som van de log-kansen van de antwoord-tokens | 2.3, 2.5 |
| `dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)` | de DPO-loss plus de impliciete beloningen | 2.5 |
| `train_reward_head(feat_w, feat_l)` | een lineaire reward-kop trainen met Adam | 2.4 |

```bash
python o2_preferences.py
# Reward-kop op synthetische data: 9x% van de testparen goed
# Alle tests geslaagd!
```

De tests controleren onder meer:
- dat bij Bradley-Terry **alleen het verschil** telt;
- dat je RM-loss niet ontploft bij een verschil van −100 (gebruik `F.logsigmoid`);
- de **verschuiving** in `sequence_logprob`: de logits op positie $t$ voorspellen het token op $t+1$;
- dat de DPO-loss bij de start ($\pi = \pi_{\text{ref}}$) precies $\ln 2 \approx 0{,}693$ is;
- dat paren die het model al goed ordent, **minder gewicht** krijgen dan paren die het fout heeft. Dat is de $\sigma(\hat r_l - \hat r_w)$ uit de gradiënt in 2.5.

<details>
<summary>💡 Hint: sequence_logprob</summary>

`torch.log_softmax(logits[:, :-1], -1)` geeft de voorspellingen voor posities 1 tot en met $T-1$. Pak met `.gather(-1, input_ids[:, 1:].unsqueeze(-1))` de log-kans van het token dat er echt stond, en vermenigvuldig met `answer_mask[:, 1:]` voordat je optelt.
</details>

<details>
<summary>💡 Hint: waarom is de DPO-loss bij de start ln 2?</summary>

Als $\pi_\theta = \pi_{\text{ref}}$ zijn beide log-verhoudingen nul, dus is het argument van de sigmoid nul, en $-\log \sigma(0) = -\log \tfrac{1}{2} = \ln 2$. Dezelfde reden waarom een binaire classifier die altijd 50% zegt een cross-entropy van $\ln 2$ heeft.
</details>

## Deel B: een echt reward model (verplicht)

```bash
python o2_preferences.py --echt
```

Dit downloadt SmolLM2-135M en het *helpful*-deel van [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), en doet wat in figuur 2.9 staat: de laatste verborgen toestand van het laatste token is de feature-vector van een gesprek, en jouw `train_reward_head` leert daar een lineaire kop op. Met 2.000 trainingsparen duurt het feature-extraheren op een CPU ongeveer 10–20 minuten.

**Beantwoord:**
1. Welk percentage van de testparen ordent je reward model goed? Vergelijk met de baseline "kies het langste antwoord" en met de 73% overeenstemming tussen menselijke labelaars bij InstructGPT. Wat zegt dat over de moeilijkheid van deze data?
2. Lees tien testparen waarin je model het **oneens** is met de mens. Zijn het fouten van het model, of twijfelachtige labels? (Tip: print `r["chosen"][-300:]` en `r["rejected"][-300:]`.)
3. Bereken de correlatie tussen de score van je reward model en de lengte van het antwoord. Wat betekent dat voor een policy die je met RL tegen dit reward model zou trainen?

## Deel C: reward hacking in het klein (aanbevolen)

Dit is het experiment van Gao et al. (2.4) in het klein, met **best-of-n** in plaats van RL. Neem 50 prompts uit de testset. Genereer per prompt 8 antwoorden met SmolLM2-135M-Instruct (temperatuur 1,0). Kies per prompt het antwoord met de hoogste score van jouw reward model.

1. Hoe lang zijn de gekozen antwoorden gemiddeld, vergeleken met alle antwoorden?
2. Lees er tien. Zijn de gekozen antwoorden echt beter, of alleen langer?
3. Bedenk een simpele correctie (bijvoorbeeld: trek $\lambda \cdot$ lengte af van de score). Wat gebeurt er?

## Deel D: DPO aan het werk (optioneel, Colab aangeraden)

Je hebt `dpo_loss` en `sequence_logprob`. Dat is alles wat je nodig hebt voor een echte DPO-run. Kijk in [`tools/m2_experiments.py`](../../../tools/m2_experiments.py) naar `exp_dpo`: dat is de run uit figuur 2.10, met LoRA uit `peft`. Vervang de loss daar door jouw `dpo_loss`, en reproduceer de figuur. Probeer daarna $\beta = 0{,}01$ en $\beta = 1$. Wat verwacht je, en wat zie je?

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je een reward model hebt getraind en kunt uitleggen waarom ongeveer 60–65% op deze data geen slechte score is (de oplossing haalt 64% met 2.000 paren);
- [ ] je de DPO-loss op een whiteboard kunt afleiden uit het KL-geregulariseerde RL-doel en Bradley-Terry.

**Oplossing:** [`../oplossingen/o2_preferences_oplossing.py`](../oplossingen/o2_preferences_oplossing.py)
