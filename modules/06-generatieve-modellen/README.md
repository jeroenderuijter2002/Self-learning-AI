# Module 6: Generatieve modellen: beeld en multimodaal

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Tekst is niet de enige modaliteit. Hoe genereert een model een foto uit ruis, en hoe 'ziet' Claude een afbeelding?

## 🌍 In de praktijk

**Stable Diffusion, 22 augustus 2022.** Begin 2022 konden alleen grote labs foto's laten genereren uit tekst (DALL·E 2 van OpenAI, achter een wachtlijst). Toen bracht Stability AI, samen met de universiteit van München en Runway, [Stable Diffusion openlijk uit](https://en.wikipedia.org/wiki/Stable_Diffusion): een diffusion-model dat op een gewone gaming-videokaart in een paar seconden een afbeelding maakt. Binnen weken bouwde de hele wereld er apps omheen. Het kan op een thuis-GPU draaien dankzij latent diffusion (hoofdstuk 6.3): het werkt niet op pixels, maar in een compacte, gecomprimeerde ruimte.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Autoencoders als concept en de transformer (module 1). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 6.1 VAEs en GANs, kort | Het idee van een latente ruimte; waarom GANs lastig te trainen zijn. |
| 6.2 Diffusion | Ruis toevoegen en leren verwijderen; de wiskunde intuïtief, inclusief de link met je Bayesiaanse achtergrond. |
| 6.3 Flow matching en moderne beeldmodellen | Latent diffusion, text-to-image, guidance. |
| 6.4 Multimodaliteit | Vision Transformers, CLIP, en hoe een LLM beeld-tokens krijgt. |

## Opdrachten en projecten (gepland)

- **Opdracht:** train een klein diffusion-model op MNIST of een eenvoudige dataset.
- **Opdracht:** gebruik CLIP voor zero-shot classificatie van eigen foto's.

## Belangrijkste bronnen

- 📖 Lilian Weng, [*What are diffusion models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 📄 [DDPM](https://arxiv.org/abs/2006.11239) · [CLIP](https://arxiv.org/abs/2103.00020)
- 🎓 [Hugging Face diffusion-cursus](https://huggingface.co/learn)

---
[← Terug naar het overzicht](../../README.md)
