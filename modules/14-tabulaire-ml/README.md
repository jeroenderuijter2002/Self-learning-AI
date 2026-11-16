# Module 14: Klassieke ML in de praktijk

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Je kent de theorie. Maar in het bedrijfsleven gaat het meeste ML-werk nog steeds over tabeldata (klanten, transacties, sensoren), en daar wint gradient boosting vaak van neurale netwerken. Dit is de praktijkkant: hoe win je echt op rommelige bedrijfsdata?

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Regressie, logistische regressie, beslisbomen, bias-variance, cross-validatie, regularisatie. Dat ken je allemaal; we beginnen bij de praktijk. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 14.1 Gradient boosting van binnen | Waarom XGBoost, LightGBM en CatBoost zo goed werken; de belangrijkste hyperparameters en hoe je ze tunet. |
| 14.2 Feature engineering | Categorische variabelen (target encoding en zijn valkuilen), tijdsfeatures, aggregaties, interacties. |
| 14.3 Data leakage | De nummer-één-fout in de praktijk: hoe het binnensluipt en hoe je het opspoort. |
| 14.4 Lastige data | Ontbrekende waarden, scheve klassen, kleine datasets, tijdsafhankelijkheid. |
| 14.5 Uitlegbaarheid | SHAP, partial dependence: uitleggen aan een klant waarom het model iets voorspelt. |
| 14.6 Wanneer wél deep learning op tabeldata | En wanneer niet. |

## Opdrachten en projecten (gepland)

- **Opdracht:** doe mee aan een Kaggle-competitie op tabeldata; doel is een plek in de top 20%.
- **Opdracht:** bouw een churn- of fraudemodel van begin tot eind, inclusief SHAP-uitleg voor een niet-technisch publiek.

## Belangrijkste bronnen

- 📄 [XGBoost](https://arxiv.org/abs/1603.02754) · [*Why do tree-based models still outperform deep learning on tabular data?*](https://arxiv.org/abs/2207.08815)
- 🎓 [Kaggle Learn](https://www.kaggle.com/learn) en winnende Kaggle-notebooks

---
[← Terug naar het overzicht](../../README.md)
