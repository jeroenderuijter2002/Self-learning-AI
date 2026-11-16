# Module 2: Van tekstvoorspeller naar assistent

⚪ *Kern: hoe moderne AI werkt* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Module 1 eindigde met een raadsel: GPT-2 en Claude hebben grofweg dezelfde architectuur, maar het ene is een tekstvoorspeller en het andere een behulpzame assistent die nadenkt. Het verschil zit volledig in de training. Na deze module kun je het hele trainingsproces van een frontier-model uitleggen.

## 🌍 In de praktijk

**Van GPT-3 naar ChatGPT (2022).** GPT-3 bestond al sinds 2020, maar was een pure tekstvoorspeller: stel het een vraag en het verzint misschien nog drie vragen erbij. Het verschil met ChatGPT zat niet in de architectuur maar in de **training uit deze module**: supervised fine-tuning plus RLHF. In het [InstructGPT-paper](https://arxiv.org/abs/2203.02155) (januari 2022) vonden mensen de antwoorden van een model van 1,3 miljard parameters, getraind met RLHF, beter dan die van het 100× grotere GPT-3 (175 miljard). Betere training versloeg brute grootte. Een paar maanden later was ChatGPT er.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Gradient descent, cross-entropy, overfitting, regularisatie. Bekend uit je ML-vakken en module 1. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 2.1 Pretraining | Wat zit er in de trainingsdata (en wat niet), datacuratie en deduplicatie, hoe een trainingsrun er in de praktijk uitziet (batchgrootte, learning-rate-schema's, verlies-spikes). |
| 2.2 Scaling laws | Waarom groter voorspelbaar beter is: Kaplan en Chinchilla. Hoe labs met een vast compute-budget kiezen tussen modelgrootte en hoeveelheid data. Emergent gedrag: echt of meetartefact? |
| 2.3 Supervised fine-tuning | Van 'maak deze webpagina af' naar 'beantwoord deze vraag': instructiedata, chat-templates, waarom een paar duizend goede voorbeelden al veel doen. |
| 2.4 RLHF | Reward models, PPO, en waarom je niet gewoon op menselijke voorkeur kunt superviseren. Reward hacking en sycophancy. |
| 2.5 Constitutional AI en DPO | Anthropic's aanpak met AI-feedback op basis van principes; Direct Preference Optimization als goedkoper alternatief voor RLHF. |
| 2.6 Reasoning-modellen | RL met verifieerbare beloningen (wiskunde, code), chain-of-thought, test-time compute: waarom 'langer nadenken' werkt. |

## Opdrachten en projecten (gepland)

- **Opdracht:** fine-tune een klein open model (bijv. met LoRA, preview van module 4) op een eigen instructiedataset en vergelijk vóór/na.
- **Opdracht:** train een mini reward model op voorkeursparen en gebruik het om antwoorden te rangschikken.
- **Experiment:** reproduceer een scaling-curve met je mini-GPT uit O3 (3–4 modelgroottes, zelfde data).

## 💼 Praktijkopdracht: Fine-tunen of prompten?

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Verzekeraar Zeeland Zeker** (fictief; denk aan een middelgrote Nederlandse verzekeraar). Ze willen een AI-assistent voor hun klantenservice. |
| **De vraag** | "Een leverancier wil ons een eigen, gefinetuned model verkopen voor een flink bedrag. Is dat nodig, of kan een bestaand model met een goede prompt het ook?" |
| **Wat je krijgt** | De [Bitext customer support-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) en een klein open model zoals [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), met [TRL](https://huggingface.co/docs/trl) voor de training. |
| **Wat je oplevert** | Een vergelijking van drie varianten (basismodel met prompt, LoRA-SFT, en SFT plus DPO), een kleine evaluatieset die je zelf opstelt, en een adviespresentatie van 5 slides. |
| **Het addertje** | Juridische zaken eist dat de assistent **nooit** een dekking toezegt die niet in de polis staat. Hoe meet je dat, en welke variant gaat er het best mee om? |
| **Wanneer is het goed?** | Je advies rust op metingen, niet op gevoel: kwaliteit, kosten en risico staan naast elkaar, en je kunt uitleggen *waarom* fine-tuning hier wel of niet helpt. |

## Belangrijkste bronnen

- 🎬 Karpathy, [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI): het hele trainingsverhaal in 3,5 uur
- 📄 [InstructGPT](https://arxiv.org/abs/2203.02155) · [Chinchilla](https://arxiv.org/abs/2203.15556) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [DPO](https://arxiv.org/abs/2305.18290)
- 💻 [nanochat](https://github.com/karpathy/nanochat): pretraining + SFT + chat in één kleine codebase
- 📖 Lilian Weng, [Lil'Log](https://lilianweng.github.io/) over RLHF en reward hacking

---
[← Terug naar het overzicht](../../README.md)
