# Module 9: Zijpaden: causal inference, GNN's, time series

🔵 *Verdieping / zijpad* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Drie onderwerpen die je profiel versterken: causal inference voor consulting ('werkt deze maatregel echt?'), graph neural networks voor netwerkdata, en time-series-modellen voor je interesse in quant en trading.

## 🌍 In de praktijk

**Google Maps, 2020.** Google Maps voorspelde aankomsttijden al voor ruim 97% van de ritten goed. Samen met DeepMind gingen ze het wegennet modelleren als een **graaf**: elk wegstuk een knoop, verbonden met de wegstukken eromheen. Met graph neural networks (hoofdstuk 9.2) werden de real-time aankomsttijden [tot 50% nauwkeuriger](https://venturebeat.com/ai/deepmind-claims-its-ai-improved-google-maps-travel-time-estimates-by-up-to-50) in steden als Berlijn, Sydney, Tokio en Washington D.C. Verkeer is van nature een netwerkprobleem: een file op de ene weg schuift door naar de volgende. Een GNN kan dat direct modelleren.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Klassieke tijdreeksanalyse (ARIMA) en regressie. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 9.1 Causal inference | Correlatie tegenover causaliteit, DAG's, confounders, difference-in-differences, uplift modelling. |
| 9.2 Graph neural networks | Message passing, GCN's, toepassingen (fraude, moleculen, supply chains). |
| 9.3 Time series met deep learning | Foundation models voor tijdreeksen, en waarom financiële data extra lastig is. |

## Opdrachten en projecten (gepland)

- **Opdracht:** schat het causale effect van een interventie op een publieke dataset, mét en zonder correctie voor confounders.
- **Opdracht:** een GNN voor node-classificatie; daarna een time-series-model op eigen crypto- of beursdata (met eerlijke backtest).

## 💼 Praktijkopdracht: Werkte de kortingsactie echt?

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Supermarktketen Buurtwinkel** (fictief; denk aan een grote Nederlandse supermarkt). Ze stuurden klanten persoonlijke kortingsbonnen. |
| **De vraag** | "Klanten die een bon kregen, gaven daarna meer uit. Marketing wil het budget verdubbelen. Is dat terecht?" |
| **Wat je krijgt** | De [Dunnhumby *The Complete Journey*-dataset](https://www.dunnhumby.com/source-files/): twee jaar aankopen van 2.500 huishoudens, inclusief campagnes en bonnen. |
| **Wat je oplevert** | Een causale analyse (bijv. propensity score matching en difference-in-differences) met een causaal diagram van je aannames, en een memo voor de marketingdirecteur. |
| **Het addertje** | De bonnen gingen vooral naar klanten die al veel kochten. Hoeveel van het 'effect' is dus gewoon selectie? |
| **Wanneer is het goed?** | Je scheidt correlatie en causaliteit expliciet, je benoemt je aannames, en je advies geeft een realistische schatting met onzekerheid. |

## Belangrijkste bronnen

- 📖 Matheus Facure, [*Causal Inference for the Brave and True*](https://matheusfacure.github.io/python-causality-handbook/) (gratis)
- 📖 Distill, [*A Gentle Introduction to Graph Neural Networks*](https://distill.pub/2021/gnn-intro/) · [GCN (Kipf & Welling)](https://arxiv.org/abs/1609.02907)

---
[← Terug naar het overzicht](../../README.md)
