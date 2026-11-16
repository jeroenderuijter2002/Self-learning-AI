# Module 10: Software engineering voor ML

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

In de praktijk is een ML engineer voor een groot deel software engineer. Sollicitaties en echte projecten stranden vaker op slordige code dan op gebrekkige ML-kennis. Deze module maakt van 'code die in mijn notebook werkt' code die een team kan onderhouden.

## 🌍 In de praktijk

**Knight Capital, 1 augustus 2012.** Een grote Amerikaanse beurshandelaar rolde nieuwe handelssoftware uit naar acht servers. Op één server werd de nieuwe code niet geïnstalleerd, en daar ging een oude, vergeten functie weer aan. Toen de beurs opende, verstuurde die server miljoenen ongewenste orders. Na 45 minuten had het bedrijf [$440 miljoen verloren](https://www.nextgov.com/digital-government/2012/08/trading-firm-lost-10-million-minute-because-computer-glitch/57188/), ongeveer $10 miljoen per minuut, en stond het aan de rand van faillissement. Geen slecht model, geen slechte strategie: slechte **software engineering**. Er was geen vaste, gecontroleerde uitrolprocedure, niemand checkte of alle servers dezelfde code draaiden, en er was geen automatische noodstop. Precies de dingen uit deze module.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Python-basis en programmeerconcepten (lussen, functies, classes). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 10.1 Python als een professional | Projectstructuur en packages, virtual environments en dependency management (uv), type hints, dataclasses en Pydantic. |
| 10.2 Testen | pytest, wat je test bij ML-code (data, transformaties, modelgedrag), fixtures, tests voor niet-deterministische code. |
| 10.3 Git en samenwerken | Branches, pull requests, code reviews, pre-commit hooks, linting (ruff). |
| 10.4 API's bouwen | Een model achter een REST-API met FastAPI: validatie, foutafhandeling, async. |
| 10.5 Containers | Docker: waarom 'het werkt op mijn machine' niet genoeg is, images, multi-stage builds. |
| 10.6 De command line en tooling | Shell, SSH, Makefiles, logging, configuratiebeheer. |

## Opdrachten en projecten (gepland)

- **Eindproject:** maak van je mini-GPT uit O3 een net Python-package met tests, een FastAPI-endpoint en een Dockerfile.
- Na deze module heeft je GitHub-profiel een project dat er professioneel uitziet.

## Belangrijkste bronnen

- 🎓 MIT, [*The Missing Semester of Your CS Education*](https://missing.csail.mit.edu) (gratis)
- 📖 Luciano Ramalho, *Fluent Python*
- 📚 [FastAPI-documentatie](https://fastapi.tiangolo.com) (een van de beste tutorials die er zijn)

---
[← Terug naar het overzicht](../../README.md)
