# Module 12: Data engineering

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

De meeste ML-projecten falen niet op het model maar op de data: die is te laat, vies, verkeerd gekoppeld of lekt informatie. Een ML engineer die zijn eigen pipelines kan bouwen, is veel waardevoller.

## 🌍 In de praktijk

**Uber Michelangelo, 2017.** Uber gebruikte machine learning voor alles: ritprijzen, fraude, aankomsttijden, de bezorgtijd van Uber Eats. Ze ontdekten dat het moeilijkste niet het model was. In hun eigen woorden: [*"building and managing data pipelines is typically one of the most costly pieces of a complete machine learning solution"*](https://www.uber.com/blog/michelangelo-machine-learning-platform/). Daarom bouwden ze een centrale **feature store** (hoofdstuk 12.6): een gedeelde, opgeschoonde verzameling van duizenden features die alle teams konden hergebruiken. Data engineering bleek het fundament onder al hun ML.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ SQL-basis (daar ben je al mee bezig) en pandas. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 12.1 SQL voor gevorderden | Window functions, CTE's, query-optimalisatie, het denken in sets. |
| 12.2 Datamodellering | Warehouses en lakehouses, star schema's, het verschil tussen OLTP en OLAP. |
| 12.3 Pipelines | ETL tegenover ELT, orkestratie (Airflow, Dagster), dbt voor transformaties. |
| 12.4 Grote data | Wanneer pandas niet meer volstaat: Polars, DuckDB, Spark in grote lijnen. |
| 12.5 Datakwaliteit | Validatie, schema's, data contracts, en vooral: data leakage voorkomen. |
| 12.6 Feature stores en streaming | Features consistent houden tussen training en productie; streaming in grote lijnen (Kafka). |

## Opdrachten en projecten (gepland)

- **Eindproject:** een pipeline die publieke data (bijv. CBS of NS) dagelijks ophaalt, opschoont, valideert en klaarzet voor een model, met orkestratie en tests.

## 💼 Praktijkopdracht: Van datamoeras naar betrouwbare cijfers

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Taxicentrale Metropool** (fictief). Elke afdeling rapporteert andere omzetcijfers uit dezelfde ritdata. |
| **De vraag** | "We willen één waarheid. Bouw een datapijplijn waarop iedereen kan vertrouwen." |
| **Wat je krijgt** | De [NYC Taxi-ritdata](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (honderden miljoenen ritten), met [DuckDB](https://duckdb.org) en dbt. |
| **Wat je oplevert** | Een pijplijn van ruwe data naar schone analysetabellen, met datakwaliteitstests, documentatie, en een featuretabel die een ML-model direct kan gebruiken. |
| **Het addertje** | Je vindt ritten met negatieve prijzen, ritten van 0 meter en ritten die eindigen vóór ze beginnen. Wat doe je ermee, en hoe leg je dat vast? |
| **Wanneer is het goed?** | Twee mensen die dezelfde vraag stellen, krijgen hetzelfde antwoord, je tests vangen slechte data, en je pijplijn draait opnieuw met één commando. |

## Belangrijkste bronnen

- 🎓 [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) (DataTalksClub, gratis)
- 📖 Martin Kleppmann, *Designing Data-Intensive Applications*
- 🔧 [DuckDB](https://duckdb.org) · [Polars](https://pola.rs)

---
[← Terug naar het overzicht](../../README.md)
