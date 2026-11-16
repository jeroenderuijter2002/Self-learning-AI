# Module 4: Inference en efficiëntie

⚪ *Kern: hoe moderne AI werkt* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Een model trainen gebeurt één keer; het draaien gebeurt miljarden keren. Kosten, snelheid en 'kan dit op onze eigen server?' bepalen in de praktijk welk model een klant kiest.

## 🌍 In de praktijk

**10 maart 2023, één avond hacken.** Een Bulgaarse ontwikkelaar, Georgi Gerganov, publiceerde *llama.cpp*: een programma om Meta's LLaMA-taalmodel op een gewone MacBook te draaien. De truc was [4-bit quantization](https://simonwillison.net/2023/Mar/11/llama/), uit hoofdstuk 4.2: het 7B-model kromp tot zo'n 4 GB. Een dag later draaide iemand het al op een Raspberry Pi. Simon Willison noemde het *"het Stable Diffusion-moment van taalmodellen"*. Sindsdien draait iedereen open modellen lokaal, ook jij met LM Studio en Ollama.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ KV-cache en GQA (module 1.4). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 4.1 Waar de tijd naartoe gaat | Prefill tegenover decode, compute-bound tegenover memory-bound, waarom de GPU vaak op geheugen wacht. |
| 4.2 Quantization | Van 16 naar 8, 4 of zelfs minder bits per gewicht: hoe het werkt (GPTQ, AWQ, GGUF) en wat je inlevert. |
| 4.3 Goedkoop fine-tunen | LoRA en QLoRA: waarom je maar 0,1% van de gewichten hoeft te trainen. |
| 4.4 Distillation | Een klein model laten leren van een groot model; zachte labels en temperatuur (Mini M1 komt terug). |
| 4.5 Serving | Batching, speculative decoding, vLLM; lokaal draaien met Ollama of llama.cpp. |

## Opdrachten en projecten (gepland)

- **Opdracht:** draai een open model lokaal in 16-, 8- en 4-bit en meet snelheid, geheugen en kwaliteit.
- **Opdracht:** LoRA-fine-tune op je laptop of Colab; vergelijk met volledige fine-tuning op een klein model.

## 💼 Praktijkopdracht: De AI-rekening moet omlaag

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord.*

| | |
|---|---|
| **De klant** | **Startup Tolkie** (fictief; denk aan een vertaal- of chat-startup). Hun product draait op een open model op gehuurde GPU's. |
| **De vraag** | "Onze GPU-rekening groeit harder dan onze omzet. Kunnen we dezelfde kwaliteit leveren voor een vijfde van de kosten?" |
| **Wat je krijgt** | Het model en de evaluatieset uit je praktijkopdracht van module 2 of 3, en een gratis Colab- of gehuurde GPU. |
| **Wat je oplevert** | Metingen van latency, throughput, geheugen en kwaliteit voor minstens drie varianten (bijv. FP16, 4-bit quantization, een gedistilleerd kleiner model), en een kostenmodel per 1.000 gebruikers. |
| **Het addertje** | De CEO wil ook dat antwoorden binnen 1 seconde beginnen (time to first token). Sommige besparingen maken dat juist slechter. |
| **Wanneer is het goed?** | Je presenteert een duidelijke afweging (kosten tegenover kwaliteit tegenover snelheid) als grafiek, en een keuze die je kunt verdedigen. |

## Belangrijkste bronnen

- 📄 [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [Distillation (Hinton)](https://arxiv.org/abs/1503.02531)
- 🎬 Karpathy, [*Let's reproduce GPT-2*](https://www.youtube.com/watch?v=l8pRSuU81PU), sectie 2 (snelheid)

---
[← Terug naar het overzicht](../../README.md)
