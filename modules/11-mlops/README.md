# Module 11: Van notebook naar productie: MLOps en LLMOps

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Een model dat in een notebook werkt is pas het begin. Bij een klant is de vraag: werkt het over een half jaar nog, kunnen we zien wanneer het slechter wordt, en kunnen we het veilig vervangen? Dat is MLOps, en precies wat consultancies voor klanten bouwen.

## 🌍 In de praktijk

**Zillow Offers, 2021.** Zillow, de grootste vastgoedsite van de VS, ging huizen kopen op basis van een algoritme dat de waarde voorspelde, om ze daarna met winst door te verkopen. Toen de huizenmarkt omsloeg, bleef het model te hoge prijzen bieden. Op 2 november 2021 stopte Zillow de hele tak: [$407,9 miljoen afgeschreven op huizen](https://www.sec.gov/Archives/edgar/data/1617640/000161764022000013/z-20211231.htm) die ze te duur hadden gekocht, en ongeveer een kwart van het personeel ontslagen. Dit is het schoolvoorbeeld van wat **MLOps** moet voorkomen: een model dat in de training goed werkte, maar waarvan niemand op tijd zag dat de wereld veranderde (data drift, hoofdstuk 11.5).

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Modellen trainen en evalueren op een testset. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 11.1 De ML-levenscyclus | Waarom ML-systemen anders falen dan gewone software; technical debt in ML. |
| 11.2 Experiment tracking en reproduceerbaarheid | MLflow of Weights & Biases, data- en modelversiebeheer, model registry. |
| 11.3 CI/CD voor modellen | Automatisch testen, trainen en uitrollen; GitHub Actions. |
| 11.4 Deployment | Batch tegenover real-time, serverless, Kubernetes in grote lijnen, één cloudplatform in de diepte (AWS, Azure of GCP). |
| 11.5 Monitoring | Data drift, concept drift, prestaties in productie meten zonder labels, alerting. |
| 11.6 LLMOps | Wat er anders is bij LLM-systemen: prompt-versiebeheer, tracing van agents, evals in productie, kosten bewaken. |

## Opdrachten en projecten (gepland)

- **Eindproject:** zet het RAG-systeem uit module 3 in productie: gecontaineriseerd, met CI/CD, tracing, monitoring en een dashboard.
- Samen met module 3 is dit je sterkste portfolioproject voor sollicitaties.

## Belangrijkste bronnen

- 🎓 [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) (DataTalksClub, gratis)
- 🎓 [Made With ML](https://madewithml.com) (Goku Mohandas, gratis) · [Full Stack Deep Learning](https://fullstackdeeplearning.com)
- 📖 Chip Huyen, *Designing Machine Learning Systems* (O'Reilly)
- 🔧 [MLflow](https://mlflow.org) · [Evidently](https://www.evidentlyai.com) (monitoring)

---
[← Terug naar het overzicht](../../README.md)
