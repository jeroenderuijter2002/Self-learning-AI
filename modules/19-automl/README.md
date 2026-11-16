# Module 19: AutoML en hyperparameter-optimalisatie

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Elk model heeft knoppen: learning rate, boomdiepte, regularisatie, architectuur. Ze met de hand of via grid search instellen is traag en duur. AutoML automatiseert dat. Een AutoML-tool aanroepen is makkelijk, en dat is precies de valkuil: je moet weten wat erachter gebeurt om te zien wanneer het misgaat. Onder de motorkap zit een mooi probleem: optimaliseren van een functie die duur is om te evalueren en waar je geen gradiënt van hebt. Daar komt je kennis van Gaussian processes en evolutionaire algoritmen samen.

## 🌍 In de praktijk

**AlphaGo, maart 2016.** Je kent AlphaGo als RL-succesverhaal (module 5). Minder bekend: DeepMind tunede de vele hyperparameters van AlphaGo herhaaldelijk met **Bayesiaanse optimalisatie**, met een Gaussian process als surrogaatmodel. Elke evaluatie was duur: een reeks partijen tegen een eerdere versie van zichzelf. Vlak voor de match tegen Lee Sedol leverde één tuningronde een flinke sprong op: de nieuwe versie [won 66,5% van de partijen tegen de oude](https://arxiv.org/abs/1812.06855), en die getunede versie speelde de slotpartij. Volgens de onderzoekers was handmatig tunen hier ondoenlijk geweest.

Het is precies het probleem van deze module: je wilt een functie optimaliseren waarvan elke meting uren kost en waarvan je geen gradiënt hebt.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Gaussian processes en Bayesiaanse inferentie (je Bayesian ML-vak), evolutionaire algoritmen, cross-validatie en hyperparameters als begrip. Je hebt AutoML al kort gezien; deze module gaat over hoe het werkt en wanneer je het (niet) moet gebruiken. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 19.1 Het probleem | Hyperparameter-optimalisatie als black-box optimalisatie; waarom random search grid search verslaat. |
| 19.2 Bayesiaanse optimalisatie | Een GP als surrogaatmodel plus een acquisitiefunctie (Expected Improvement, UCB): exploration tegenover exploitation. TPE, de variant in Optuna. |
| 19.3 Goedkoper zoeken | Multi-fidelity: Successive Halving en Hyperband stoppen slechte kandidaten vroeg. Early stopping en warm-starting. |
| 19.4 Neural architecture search | Architecturen zoeken met RL en evolutionaire algoritmen, en waarom NAS grotendeels uit de mode raakte. |
| 19.5 AutoML-systemen | AutoGluon, auto-sklearn, H2O: hoe ze ensembles stapelen, en waarom ze vaak een sterke baseline voor tabeldata zijn. |
| 19.6 Grenzen | Overfitting op je validatieset, rekenkosten, en wat AutoML niet doet: het probleem formuleren, features bedenken, data lekken voorkomen (module 14). |

## Opdrachten en projecten (gepland)

- **Opdracht:** implementeer Bayesiaanse optimalisatie met een GP en Expected Improvement vanaf nul, en vergelijk met random search en Optuna op een echt model.
- **Opdracht:** zet AutoGluon tegenover je eigen, met de hand getunede gradient boosting uit module 14. Wie wint, en wat kost het?

## 💼 Praktijkopdracht: Kunnen we ons data science-team vervangen door AutoML?

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Verzekeraar Polderpolis** (fictief). De CFO las dat AutoML 'net zo goed is als een data scientist'. |
| **De vraag** | "Kunnen we bezuinigen op ons team en alles met AutoML doen? Wij willen een eerlijk antwoord, geen verkooppraatje." |
| **Wat je krijgt** | Vijf tot tien datasets van [OpenML](https://www.openml.org), [AutoGluon](https://auto.gluon.ai), en je eigen getunede modellen uit module 14. |
| **Wat je oplevert** | Een benchmark van AutoML tegenover handwerk (kwaliteit, rekentijd, kosten), plus een analyse van wat AutoML níet doet, en een memo voor de CFO. |
| **Het addertje** | Op de benchmark wint AutoML vaak. Maar de lastigste fouten in de praktijk (verkeerde doelvariabele, datalekken) zitten vóór de modelkeuze. Hoe laat je dat zien? |
| **Wanneer is het goed?** | Je memo is genuanceerd en concreet: waar AutoML het team versterkt, waar het tekortschiet, met cijfers uit je eigen benchmark. |

## Belangrijkste bronnen

- 📖 Hutter, Kotthoff & Vanschoren (red.), [*Automated Machine Learning*](https://www.automl.org/book/) (gratis, open access)
- 📄 [Brochu et al., tutorial Bayesiaanse optimalisatie](https://arxiv.org/abs/1012.2599) · [Hyperband](https://arxiv.org/abs/1603.06560) · [NAS met RL](https://arxiv.org/abs/1611.01578)
- 🔧 [Optuna](https://optuna.org) · [AutoGluon](https://auto.gluon.ai)

---
[← Terug naar het overzicht](../../README.md)
