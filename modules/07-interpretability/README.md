# Module 7: Interpretability en alignment

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Wat gebeurt er écht in de gewichten? In module 1 zag je al previous-token heads, induction heads en de logit lens. Dit is de volledige versie van dat onderzoek, plus de vraag hoe je zorgt dat modellen doen wat je bedoelt.

## 🌍 In de praktijk

**Golden Gate Claude, mei 2024.** Onderzoekers van Anthropic trainden een sparse autoencoder (hoofdstuk 7.3) op de binnenkant van Claude 3 Sonnet, en vonden miljoenen herkenbare "features": één ervan ging aan bij de Golden Gate Bridge. Daarna draaiden ze die ene feature handmatig hoog. Het resultaat: [Claude dacht dat hij de Golden Gate Bridge *was*](https://www.anthropic.com/news/golden-gate-claude), en begon over de brug in vrijwel elk antwoord. Ze zetten het 24 uur online als demo. Het was het eerste overtuigende bewijs dat je in een groot, echt taalmodel kunt **zien** waar een begrip zit, en het kunt **sturen**.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Attention heads, residual stream, logit lens (module 1). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 7.1 Circuits | Induction heads en het indirect-object-circuit uit O4, nu volledig ontleed. |
| 7.2 Superposition | Waarom één neuron vele begrippen tegelijk codeert. |
| 7.3 Sparse autoencoders | Begrippen ('features') uit een model halen en ze aan- of uitzetten. |
| 7.4 Alignment-problemen | Reward hacking, sycophancy, misgeneralisatie, en hoe je ze test (red-teaming, evals). |

## Opdrachten en projecten (gepland)

- **Opdracht:** vind en ablateer de induction heads in GPT-2 met TransformerLens.
- **Opdracht:** train een kleine sparse autoencoder op GPT-2-activaties en interpreteer een paar features.

## Belangrijkste bronnen

- 📄 Anthropic, [transformer-circuits.pub](https://transformer-circuits.pub), o.a. [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html)
- 🎓 ARENA-curriculum (interpretability-hoofdstuk), TransformerLens-tutorials

---
[← Terug naar het overzicht](../../README.md)
