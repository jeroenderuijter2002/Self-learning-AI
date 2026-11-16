# Module 15: Self-supervised en representatie-leren

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

Gelabelde data is duur; ongelabelde data is er in overvloed. Self-supervised learning laat een model zijn eigen leertaak maken uit ruwe data, zoals 'voorspel het volgende woord' of 'welk bijschrift hoort bij deze foto?'. Daarmee leert het bruikbare representaties (embeddings) zonder dat een mens iets labelt. Het is het geheime ingrediënt onder GPT, BERT, CLIP en vrijwel elk foundation model.

## 🌍 In de praktijk

**CLIP, januari 2021.** Een beeldmodel trainen kostte altijd een gigantische gelabelde dataset: ImageNet heeft 1,28 miljoen foto's die mensen met de hand hebben gelabeld. OpenAI pakte het anders aan met CLIP: een model dat alleen leerde welk bijschrift bij welke foto hoort, op [400 miljoen foto-tekstparen van internet](https://arxiv.org/abs/2103.00020). Niemand labelde iets; de bijschriften wáren de supervisie. Het resultaat: CLIP haalde op ImageNet dezelfde nauwkeurigheid als het klassieke ResNet-50, **zonder ook maar één van die 1,28 miljoen trainingsvoorbeelden te gebruiken**. CLIP zit nu in beeldzoekmachines, in Stable Diffusion (module 6), en in de manier waarop taalmodellen leren 'kijken'.

En de grootste self-supervised modellen van allemaal? Dat zijn de taalmodellen zelf: "voorspel het volgende woord" is een leertaak die je uit elke tekst gratis kunt halen.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ Supervised learning, autoencoders als concept, embeddings (module 1.1) en de pretraining-taak van taalmodellen (module 2). Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 15.1 Waarom leren zonder labels? | Het labelprobleem, en het idee van een 'pretext task': een taak die je zelf uit de data maakt. |
| 15.2 Contrastive learning | SimCLR en de InfoNCE-loss: trek twee versies van dezelfde foto naar elkaar toe, duw andere foto's weg. |
| 15.3 Masked modelling | Stukjes verbergen en laten reconstrueren: BERT voor tekst, MAE voor beeld. |
| 15.4 Zelf-distillatie | DINO en DINOv2: een model leert van een iets oudere versie van zichzelf, zonder negatieve voorbeelden. |
| 15.5 Tekst en beeld samen | CLIP: miljoenen foto's met bijschriften als gratis supervisie, en zero-shot classificatie. |
| 15.6 Representaties gebruiken | Linear probing, fine-tuning, en embeddings als basis voor zoeken en clusteren (link met module 3). |

## Opdrachten en projecten (gepland)

- **Opdracht:** implementeer SimCLR op CIFAR-10 en meet met een linear probe hoe goed de representaties zijn, vergeleken met een supervised model.
- **Opdracht:** gebruik CLIP voor zero-shot classificatie en zoeken in een eigen fotocollectie ("vind alle foto's van een fiets in de sneeuw").

## Belangrijkste bronnen

- 📄 [SimCLR](https://arxiv.org/abs/2002.05709) · [DINO](https://arxiv.org/abs/2104.14294) · [MAE](https://arxiv.org/abs/2111.06377) · [CLIP](https://arxiv.org/abs/2103.00020)
- 📖 Lilian Weng, [*Self-Supervised Representation Learning*](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)

---
[← Terug naar het overzicht](../../README.md)
