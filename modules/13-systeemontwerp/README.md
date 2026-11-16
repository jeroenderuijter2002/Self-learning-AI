# Module 13: ML-systeemontwerp en evaluatie

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

'Ontwerp een aanbevelingssysteem voor een streamingdienst' is een standaardvraag bij sollicitaties voor ML engineers, en de dagelijkse realiteit bij een consultancy. Het gaat om de keuzes rond het model: wat optimaliseer je, met welke data, hoe meet je succes, wat kan er misgaan?

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Losse modeltypen en standaardmetrics (accuracy, precision/recall, AUC). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 13.1 Het raamwerk | Van bedrijfsdoel naar ML-probleem: doelvariabele, metrics, randvoorwaarden (latency, kosten, privacy). |
| 13.2 Evaluatie die ertoe doet | Offline tegenover online metrics, de juiste validatiestrategie (tijdsgebonden splits), kalibratie, bedrijfsmetrics. |
| 13.3 Experimenteren | A/B-testen, power-analyse, valkuilen (peeking, netwerkeffecten), multi-armed bandits. |
| 13.4 Klassieke systemen | Aanbevelingen, zoeken en ranking, fraudedetectie, vraagvoorspelling: de standaardarchitecturen. |
| 13.5 LLM-systemen ontwerpen | Wanneer wel en niet een LLM, kosten-batenafweging, guardrails, human-in-the-loop. |
| 13.6 Oefencases | Uitgewerkte casussen in sollicitatieformat, om hardop te oefenen. |

## Opdrachten en projecten (gepland)

- **Opdracht:** werk 6 systeemontwerp-cases schriftelijk uit en laat ze door Claude kritisch beoordelen als 'interviewer'.
- **Opdracht:** analyseer een A/B-test op een publieke dataset, inclusief power-berekening.

## Belangrijkste bronnen

- 📖 Chip Huyen, *Designing Machine Learning Systems*
- 📖 Ali Aminian & Alex Xu, *Machine Learning System Design Interview*
- 📖 Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (het standaardwerk over A/B-testen)

---
[← Terug naar het overzicht](../../README.md)
