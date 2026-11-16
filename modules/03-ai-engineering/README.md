# Module 3: AI engineering: LLM's in systemen

⚪ *Kern: hoe moderne AI werkt* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Dit is wat bedrijven nu het meest bouwen en wat consultancies als QuantumBlack en BCG X verkopen: systemen rond een LLM. Het model is een onderdeel; het systeem eromheen bepaalt of het werkt.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Hoe een LLM intern werkt (module 1). Basis-API-gebruik. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 3.1 Prompting serieus genomen | System prompts, few-shot, structured output (JSON-schema's), waarom prompts in feite programma's zijn. |
| 3.2 Embeddings en vector search | Van de embeddings uit 1.1 naar zinsembeddings; cosine similarity, vector databases, approximate nearest neighbour (HNSW). |
| 3.3 RAG | Retrieval-augmented generation: chunking, hybrid search (BM25 + vectoren), reranking, en de vijf manieren waarop RAG in de praktijk misgaat. |
| 3.4 Tool use en agents | Function calling, de agent-loop, wanneer een simpele workflow beter is dan een agent, MCP als standaard om tools aan te sluiten. |
| 3.5 Evals | Het belangrijkste en meest onderschatte onderdeel: testsets bouwen, LLM-as-judge, regressietests voor prompts. 'Als je het niet meet, weet je niet of het werkt.' |

## Opdrachten en projecten (gepland)

- **Eindproject:** een RAG-systeem over een eigen documentverzameling (bijv. TU/e-studiemateriaal), mét evaluatieset en gemeten verbeteringen per iteratie.
- **Opdracht:** bouw een kleine agent met 3 tools en schrijf evals voor zijn faalgevallen.
- Dit project zet je in module 11 in productie: samen een sterk portfolio-item.

## Belangrijkste bronnen

- 📖 Chip Huyen, *AI Engineering* (O'Reilly, 2025)
- 📄 [RAG (Lewis et al.)](https://arxiv.org/abs/2005.11401) · Anthropic, [*Building effective agents*](https://www.anthropic.com/research/building-effective-agents)
- 🔌 [Model Context Protocol](https://modelcontextprotocol.io)
- 🎓 [Hugging Face Learn](https://huggingface.co/learn): LLM- en agents-cursussen

---
[← Terug naar het overzicht](../../README.md)
