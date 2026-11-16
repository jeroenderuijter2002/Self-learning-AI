# Module 20: Recommender systems

🔵 *Verdieping / zijpad* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Aanbevelingssystemen zijn waarschijnlijk de ML-toepassing met de meeste commerciële waarde: ze bepalen wat je ziet op Netflix, Spotify, YouTube, Amazon en bol. De basisversie (matrixfactorisatie) is verrassend simpel. De industriële versie is een van de complexste ML-systemen die er bestaan: miljoenen items, nauwelijks expliciete feedback, nieuwe gebruikers zonder geschiedenis, en een model dat zijn eigen trainingsdata beïnvloedt. Voor consultancy (retail, e-commerce) en sollicitatiecases is dit standaardkennis.

## 🌍 In de praktijk

**Netflix: $1 miljoen voor een algoritme dat nooit live ging.** In 2006 loofde Netflix een miljoen dollar uit voor wie zijn aanbevelingssysteem 10% nauwkeuriger kon maken. In september 2009 won het team BellKor's Pragmatic Chaos met [een verbetering van 10,06%](https://thenextweb.com/news/remember-netflixs-1m-algorithm-contest-well-heres-why-it-didnt-use-the-winning-entry/). Het winnende ensemble van honderden modellen is daarna nooit in productie genomen: volgens Netflix woog de extra nauwkeurigheid niet op tegen het engineeringwerk, en het bedrijf was inmiddels overgestapt van dvd's naar streaming. Een les die je in module 13 terugziet.

Hoeveel recommenders waard zijn, schreven twee Netflix-topmensen in 2015: het aanbevelingssysteem bepaalt de keuze voor [zo'n 80% van de gestreamde uren](https://doi.org/10.1145/2843948), en personalisatie en aanbevelingen samen besparen Netflix naar eigen schatting **meer dan $1 miljard per jaar**, vooral doordat minder mensen opzeggen.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Matrixfactorisatie als lineaire algebra (SVD), embeddings (module 1.1) en de basis van classificatie en ranking-metrics. Je hebt recommenders al kort gezien; deze module gaat naar de moderne, industriële aanpak. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 20.1 Het probleem | Expliciete tegenover impliciete feedback, de long tail, en waarom 'voorspel de rating' niet het echte doel is. |
| 20.2 Collaborative filtering | Item-item-similariteit en matrixfactorisatie: de technieken van de Netflix Prize. |
| 20.3 Deep learning-recommenders | Two-tower-modellen voor retrieval, en ranking-modellen met veel features (zoals bij YouTube). |
| 20.4 De twee-traps-architectuur | Snel kandidaten ophalen (approximate nearest neighbours, link met vector search in module 3), daarna nauwkeurig ranken. |
| 20.5 Evaluatie | Offline metrics (recall@k, NDCG) tegenover online A/B-tests (module 13), en waarom ze het vaak oneens zijn. |
| 20.6 Valkuilen | Cold start, feedback loops en popularity bias, filterbubbels, en exploration via bandits (link met module 5). |

## Opdrachten en projecten (gepland)

- **Opdracht:** bouw op MovieLens een recommender in drie stappen: populariteit, matrixfactorisatie en een two-tower-model. Vergelijk ze op recall@10 en op hoe gevarieerd de aanbevelingen zijn.
- **Opdracht:** simuleer een feedback loop: laat je model steeds trainen op klikken die het zelf heeft veroorzaakt, en meet hoe de aanbevelingen versmallen.

## 💼 Praktijkopdracht: Aanbevelingen voor een streamingdienst

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Streamingdienst Kijkdoos** (fictief; denk aan Netflix of Videoland). Ze willen kijkers langer vasthouden met betere aanbevelingen. |
| **De vraag** | "Bouw een aanbevelingssysteem dat beter is dan 'meest bekeken'. En zorg dat nieuwe abonnees ook meteen iets goeds zien." |
| **Wat je krijgt** | [MovieLens](https://grouplens.org/datasets/movielens/) (miljoenen echte filmbeoordelingen). |
| **Wat je oplevert** | Een recommender in stappen (populariteit, matrixfactorisatie, two-tower), offline evaluatie (recall@k, diversiteit, dekking), een cold-start-oplossing, en een A/B-testplan voor de lancering. |
| **Het addertje** | De contentafdeling wil dat eigen Nederlandse producties vaker worden getoond, ook als het model ze lager inschat. Hoe verwerk je dat, en wat kost het? |
| **Wanneer is het goed?** | Je verslaat de populariteits-baseline aantoonbaar, je cold-start-aanpak werkt, en je maakt de afweging tussen kijkersgedrag en bedrijfsdoelen expliciet. |

## Belangrijkste bronnen

- 📄 Koren, Bell & Volinsky, [*Matrix Factorization Techniques for Recommender Systems*](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) (2009, van het winnende Netflix Prize-team)
- 📄 Covington et al., [*Deep Neural Networks for YouTube Recommendations*](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) (2016): het klassieke twee-traps-ontwerp
- 📄 Gomez-Uribe & Hunt, [*The Netflix Recommender System*](https://doi.org/10.1145/2843948) (2015)
- 🔧 Microsoft [recommenders](https://github.com/recommenders-team/recommenders): implementaties en notebooks van de belangrijkste algoritmes

---
[← Terug naar het overzicht](../../README.md)
