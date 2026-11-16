# Module 18: GPU-programmering en distributed training

🟠 *Engineering: van model naar product* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Moderne AI is in de kern een rekenprobleem: wie de hardware het slimst benut, traint goedkoper en sneller. Deze module gaat onder de motorkap van PyTorch: hoe werkt een GPU, waarom is geheugen vaak de echte bottleneck, en hoe verdeel je één model over honderden GPU's? Belangrijk bij AI-labs en bij iedereen die zelf grote modellen traint of draait.

## 🌍 In de praktijk

**DeepSeek-V3, december 2024.** Een Chinees lab bracht een open taalmodel uit dat volgens hun eigen technische rapport kon meekomen met de beste gesloten modellen. Wat de wereld schokte, was de prijs: de volledige training kostte [2,788 miljoen GPU-uren op H800-chips, ongeveer $5,6 miljoen](https://arxiv.org/abs/2412.19437) (bij een huurprijs van $2 per GPU-uur). Dat is een fractie van wat men voor zo'n model verwachtte. Het geheim zat voor een groot deel in precies deze module: trainen in **FP8** (hoofdstuk 18.3), slim verdelen over 2.048 GPU's met een eigen pipeline-algoritme (18.5), en het geheugen zo optimaliseren dat dure tensor-parallelisme niet nodig was. Het rapport laat zien dat engineering op hardwareniveau net zo belangrijk is als het model zelf.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ PyTorch-basis, matrixvermenigvuldiging (Mini M3), en de inference-optimalisaties uit module 4 (quantization, KV-cache). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 18.1 Hoe een GPU werkt | Duizenden simpele rekenkernen, het geheugenhiërarchie-probleem, en waarom de GPU vaak op data wacht in plaats van rekent. |
| 18.2 Je eerste kernels | CUDA-basis en Triton: een kernel schrijven in Python; een matrixvermenigvuldiging en softmax die sneller zijn dan de naïeve versie. |
| 18.3 Mixed precision | FP32, BF16, FP8: minder bits, sneller rekenen, en hoe je voorkomt dat training instabiel wordt. |
| 18.4 Data- en modelparallelisme | DDP, FSDP en ZeRO: wanneer kopieer je het model, wanneer knip je het op? |
| 18.5 Tensor- en pipelineparallelisme | Hoe modellen van honderden miljarden parameters over duizenden GPU's verdeeld worden (Megatron-LM). |
| 18.6 Profileren en kosten | Meten waar de tijd heen gaat, Model FLOPs Utilization (MFU), en wat een trainingsrun echt kost. |

## Opdrachten en projecten (gepland)

- **Opdracht:** schrijf een fused softmax-kernel in Triton en meet de versnelling tegenover PyTorch.
- **Opdracht:** train je mini-GPT uit O3 met DDP over meerdere GPU's (Colab of een gehuurde cloud-GPU) en meet de schaalbaarheid.

## Belangrijkste bronnen

- 📖 Hugging Face, [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook): gratis, interactief, over trainen op duizenden GPU's
- 🔧 [Triton](https://triton-lang.org) · Karpathy's [llm.c](https://github.com/karpathy/llm.c) (GPT-2 trainen in pure C/CUDA)
- 📄 [ZeRO](https://arxiv.org/abs/1910.02054) · [Megatron-LM](https://arxiv.org/abs/1909.08053) · [DeepSeek-V3 technisch rapport](https://arxiv.org/abs/2412.19437)
- 📖 Kirk & Hwu, *Programming Massively Parallel Processors*

---
[← Terug naar het overzicht](../../README.md)
