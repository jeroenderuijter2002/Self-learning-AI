# Module 12: Data engineering

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

De meeste ML-projecten falen niet op het model maar op de data: die is te laat, vies, verkeerd gekoppeld of lekt informatie. Een ML engineer die zijn eigen pipelines kan bouwen, is veel waardevoller.

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

## Belangrijkste bronnen

- 🎓 [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) (DataTalksClub, gratis)
- 📖 Martin Kleppmann, *Designing Data-Intensive Applications*
- 🔧 [DuckDB](https://duckdb.org) · [Polars](https://pola.rs)

---
[← Terug naar het overzicht](../../README.md)
