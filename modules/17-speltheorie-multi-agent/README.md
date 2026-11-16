# Module 17: Speltheorie en multi-agent systemen

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Zodra er meerdere beslissers zijn, verandert alles: wat voor jou optimaal is, hangt af van wat de ander doet. Denk aan veilingen, onderhandelingen, verkeer, financiële markten, en steeds vaker: meerdere AI-agents die samenwerken of concurreren. Deze module verbindt klassieke speltheorie met multi-agent reinforcement learning.

## 🌍 In de praktijk

**Pluribus, juli 2019.** AI had al gewonnen met dammen, schaken, Go en poker tegen één tegenstander. Maar al die spellen hebben twee spelers, en wat de één wint verliest de ander. Poker met zes spelers is van een andere orde: geen enkele strategie is daar gegarandeerd optimaal. Toch versloeg Pluribus, gemaakt door Carnegie Mellon en Facebook AI, [topprofessionals in six-player no-limit Texas hold'em](https://www.sciencedaily.com/releases/2019/07/190711141343.htm). Onder de verslagen spelers zat onder anderen Darren Elias. Pluribus leerde door tegen zichzelf te spelen, en zijn kracht zat in het onvoorspelbaar mengen van strategieën: precies het **gemengde strategieën**-idee uit de speltheorie.

**OpenAI Five, april 2019.** In hetzelfde jaar versloegen vijf samenwerkende AI-agents de regerend wereldkampioenen van het teamspel Dota 2, [Team OG, met 2–0](https://openai.com/index/openai-five-defeats-dota-2-world-champions/): de eerste keer dat AI wereldkampioenen versloeg in een e-sport. Vijf agents, één team: multi-agent RL op enorme schaal.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Single-agent RL en MDP's (module 5 en je eigen achtergrond) en optimalisatie. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 17.1 Speltheorie in vogelvlucht | Normaalvorm, Nash-evenwicht, gemengde strategieën, zero-sum tegenover algemene spellen. |
| 17.2 Spellen met onvolledige informatie | Poker als testbed: counterfactual regret minimization (CFR), de techniek achter Pluribus. |
| 17.3 Multi-agent RL | Waarom de omgeving 'niet-stationair' wordt als iedereen tegelijk leert; centralized training, decentralized execution. |
| 17.4 Self-play en populaties | Leren door tegen jezelf en tegen een hele populatie te spelen (AlphaStar, OpenAI Five). |
| 17.5 Mechanism design | De omgekeerde vraag: hoe ontwerp je de regels (bijv. een veiling) zodat het gewenste gedrag ontstaat? |
| 17.6 Multi-agent LLM-systemen | Meerdere LLM-agents laten samenwerken, debatteren of onderhandelen; link met module 3. |

## Opdrachten en projecten (gepland)

- **Opdracht:** implementeer CFR voor Kuhn poker (een poker-variant met 3 kaarten) en zie hoe het naar het Nash-evenwicht convergeert.
- **Opdracht:** train meerdere agents in een PettingZoo-omgeving en vergelijk onafhankelijk leren met gedeelde training.

## Belangrijkste bronnen

- 📖 Albrecht, Christianos & Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024; gratis online)
- 📄 [Pluribus (Science, 2019)](https://www.science.org/doi/10.1126/science.aay2400)
- 🔧 [PettingZoo](https://pettingzoo.farama.org): de standaardbibliotheek voor multi-agent omgevingen

---
[← Terug naar het overzicht](../../README.md)
