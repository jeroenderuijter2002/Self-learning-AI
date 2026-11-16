# Module 8: Klassieke en symbolische AI

🔵 *Verdieping / zijpad* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

'AI' is ouder en breder dan machine learning. Zoeken en plannen komen terug in moderne agents, waar een LLM als heuristiek dient.

## 🌍 In de praktijk

**Deep Blue tegen Kasparov, 11 mei 1997.** IBM's schaakcomputer Deep Blue versloeg wereldkampioen Garry Kasparov met 3½–2½: de eerste keer dat een computer een regerend wereldkampioen versloeg [in een match onder normale toernooiregels](https://www.guinnessworldrecords.com/world-records/100873-first-computer-to-beat-a-world-chess-champion-under-regular-time-controls). Er zat vrijwel geen machine learning in, vooral **zoeken** (hoofdstuk 8.1), met speciale chips die zo'n 200 miljoen posities per seconde bekeken, plus handgeschreven schaakkennis.

Vergelijk dat met AlphaGo in 2016 (module 5): bij Go werkt puur zoeken niet meer, daar was leren nodig. En moderne agents combineren beide: een LLM als "intuïtie", plus zoeken en plannen.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Optimalisatie en LP/MIP (je OR-achtergrond). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 8.1 Zoeken | BFS/DFS, A*, heuristieken; spellen met minimax. |
| 8.2 Constraint satisfaction en planning | CSP's, STRIPS-achtige planning. |
| 8.3 Logica en kennisrepresentatie | Propositie- en predicatenlogica, knowledge graphs. |
| 8.4 Neuro-symbolisch | LLM's combineren met solvers, zoeken en verificatie. |

## Opdrachten en projecten (gepland)

- **Opdracht:** A* voor een routeplanner; daarna een LLM als heuristiek inzetten en vergelijken.

## Belangrijkste bronnen

- 📖 Russell & Norvig, [*Artificial Intelligence: A Modern Approach*](https://aima.cs.berkeley.edu)

---
[← Terug naar het overzicht](../../README.md)
