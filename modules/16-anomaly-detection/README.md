# Module 16: Anomaly detection

🔵 *Verdieping / zijpad* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Fraude, defecte machines, hackers, afwijkende sensordata: in de praktijk wil je heel vaak het zeldzame vinden. Het lastige is dat je van 'normaal' miljoenen voorbeelden hebt en van 'fout' bijna geen. Gewone classificatie werkt dan slecht. Deze module behandelt de technieken die wél werken, en hoe je ze eerlijk evalueert.

## 🌍 In de praktijk

**Visa, elke transactie, in een milliseconde.** Visa verwerkte in 2018 meer dan 127 miljard transacties, en liet [AI op 100% daarvan los](https://www.pymnts.com/visa/2019/ai-fraud-prevention-advanced-authorization/): elke betaling wordt in ongeveer een milliseconde beoordeeld op meer dan 500 risicokenmerken. Volgens Visa voorkwamen banken daarmee zo'n **$25 miljard aan fraude per jaar**. Later meldde Visa dat het in één jaar [80 miljoen frauduleuze transacties ter waarde van $40 miljard](https://www.pymnts.com/news/artificial-intelligence/2024/visa-ai-helped-block-80-million-fraudulent-transactions-in-2023/) had tegengehouden.

Het lastige, en precies het onderwerp van deze module: verreweg de meeste transacties zijn gewoon in orde. Een model dat alles goedkeurt, scoort dus 99,9% "accuracy" en is volkomen waardeloos. En elke onterecht geblokkeerde kaart is een boze klant.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Classificatie, clustering, PCA en dichtheidsschatting: die ken je al. We bouwen daarop voort. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 16.1 Het probleem | Soorten afwijkingen (punt, context, collectief), extreem scheve klassen, en waarom accuracy hier liegt. |
| 16.2 Statistisch en op afstand | Z-scores, Mahalanobis-afstand, k-nearest neighbours, Local Outlier Factor. |
| 16.3 Isolation Forest en one-class modellen | Afwijkingen zijn makkelijk te isoleren; one-class SVM. |
| 16.4 Deep anomaly detection | Autoencoders (een hoge reconstructiefout is verdacht), en self-supervised representaties (module 15). |
| 16.5 Tijdreeksen | Afwijkingen in sensordata en transacties over tijd; seizoenspatronen niet verwarren met afwijkingen. |
| 16.6 In productie | Weinig labels, drift, alert-moeheid, de mens in de loop, en de juiste metrics (precision@k, PR-AUC). |

## Opdrachten en projecten (gepland)

- **Opdracht:** fraudedetectie op de bekende creditcarddataset van de ULB (0,17% fraude): vergelijk Isolation Forest, een autoencoder en gradient boosting.
- **Opdracht:** afwijkingen vinden in een sensortijdreeks, inclusief een eerlijke evaluatie op labels die je pas achteraf gebruikt.

## 💼 Praktijkopdracht: Afwijkingen in de chipfabriek

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Een halfgeleiderfabrikant** (fictief; denk aan een chipfabriek). Elke defecte wafer die laat ontdekt wordt, kost veel geld. |
| **De vraag** | "We meten honderden sensorsignalen per productiestap, maar zien pas aan het eind dat iets mis is. Kunnen we eerder ingrijpen?" |
| **Wat je krijgt** | De [SECOM-dataset](https://archive.ics.uci.edu/dataset/179/secom): 590 sensorsignalen uit een echte halfgeleiderproductie, met een klein aantal mislukte producten. |
| **Wat je oplevert** | Een vergelijking van minstens drie methoden (bijv. Isolation Forest, een autoencoder, een supervised model), een kostenafweging (een vals alarm stopt de lijn, een gemiste fout kost een wafer), en een advies. |
| **Het addertje** | De data heeft veel ontbrekende waarden en maar een handvol fouten. Hoe evalueer je eerlijk zonder jezelf voor de gek te houden? |
| **Wanneer is het goed?** | Je kiest je drempel op basis van de kosten van fouten, je evaluatie houdt rekening met de scheve klassen, en je advies is uitvoerbaar op de werkvloer. |

## Belangrijkste bronnen

- 🔧 [PyOD](https://pyod.readthedocs.io): een Python-bibliotheek met tientallen anomaly-detectie-algoritmes
- 📊 [Credit Card Fraud-dataset (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) voor tijdreeksen

---
[← Terug naar het overzicht](../../README.md)
