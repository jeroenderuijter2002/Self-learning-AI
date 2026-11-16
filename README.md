# Zelf AI aanleren
### Van ML-basis naar hoe moderne AI (zoals Claude) écht werkt
*Zelfstudie-cursus voor Jeroen — editie 2026-2027*

---

## Cursusdoelen

Je kent machine learning: optimalisatie, generalisatie, probabilistische modellen, Bayesiaanse inferentie. Deze cursus bouwt daarop verder, op het terrein waar de meeste ML-opleidingen stoppen: **de technologie achter huidige AI-systemen**.

Na deze cursus kun je:
- uitleggen, tot op matrixniveau, wat er gebeurt als een taalmodel een woord voorspelt;
- beschrijven hoe een model als Claude wordt getraind: pretraining, post-training en RL;
- zelf LLM-systemen bouwen en evalueren (RAG, agents, tools);
- de bredere AI-wereld plaatsen: RL, generatieve beeldmodellen, interpretability en klassieke AI.

**Filosofie:** zelf bouwen gaat boven lezen. Elke module heeft uitleg, video's en opdrachten waarin je dingen vanaf nul implementeert. Een module is een hoofdstuk, geen collegeblok. Werk in je eigen tempo.

---

## Modules

| # | Module | Status | Notes | Opdrachten | Video's |
|---|--------|--------|-------|------------|---------|
| ⚪ 1 | **Transformers van binnen** | ✅ beschikbaar | [README](modules/01-transformers/README.md) · [1.1](modules/01-transformers/notes/1.1-tokens-en-embeddings.md) · [1.2](modules/01-transformers/notes/1.2-attention.md) · [1.3](modules/01-transformers/notes/1.3-transformer-block.md) · [1.4](modules/01-transformers/notes/1.4-van-model-naar-tekst.md) | [O1](modules/01-transformers/opdrachten/O1-tokenizer.md) · [O2](modules/01-transformers/opdrachten/O2-attention-numpy.md) · [O3](modules/01-transformers/opdrachten/O3-bouw-je-eigen-gpt.md) · [O4](modules/01-transformers/opdrachten/O4-gpt2-onder-de-loep.md) | [Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · [3B1B](https://www.youtube.com/watch?v=eMlx5fFNoYc) |
| ⚪ 2 | Hoe een model als Claude gemaakt wordt (pretraining, RLHF, Constitutional AI, reasoning) | 🔜 | | | |
| ⚪ 3 | AI engineering: RAG, tools, agents, evals | 🔜 | | | |
| ⚪ 4 | Inference & efficiëntie: quantization, LoRA, distillation | 🔜 | | | |
| 🟢 5 | Reinforcement learning | 🔜 | | | |
| 🟢 6 | Generatieve modellen: diffusion, multimodaal | 🔜 | | | |
| 🟢 7 | Interpretability & alignment | 🔜 | | | |
| 🔵 8 | Klassieke & symbolische AI | 🔜 | | | |
| 🔵 9 | Zijpaden: causal inference, GNN's, time-series | 🔜 | | | |

⚪ = kern (hoe moderne AI werkt) · 🟢 = verbreding · 🔵 = verdieping/zijpad

**Aanbevolen volgorde:** 1 → 2 → 3, daarna vrij kiezen.

---

## Mini's

Korte deep-dives over één concept dat steeds terugkomt.

| Mini | Onderwerp |
|------|-----------|
| 🟡 M1 | [Softmax en temperatuur](minis/M1-softmax-en-temperatuur.md) |
| 🟡 M2 | [Waarom delen door √d_k?](minis/M2-waarom-wortel-dk.md) |

---

## Materiaal

**Software:** zie [SOFTWARE.md](SOFTWARE.md). Python + NumPy + PyTorch is genoeg. Google Colab werkt ook.

**Kernbronnen** (allemaal gratis):
- Andrej Karpathy, *[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)*: de beste "bouw het zelf"-serie die er is
- 3Blue1Brown, *[Deep Learning-serie](https://www.3blue1brown.com/topics/neural-networks)*: visuele intuïtie
- Stanford CS336, *[Language Modeling from Scratch](https://stanford-cs336.github.io/)*: universitair niveau, voor als je dieper wilt
- Lilian Weng, *[Lil'Log](https://lilianweng.github.io/)*: heldere overzichtsposts over vrijwel elk onderwerp

---

## Repostructuur

```
README.md               ← je bent hier
SOFTWARE.md             ← installatie
minis/                  ← korte deep-dives
modules/
  01-transformers/
    README.md           ← leerdoelen, leespad, planning
    notes/              ← uitleg (lees deze in volgorde)
    opdrachten/         ← opdrachten + Python-skeletons
    oplossingen/        ← pas kijken als je vastzit
    zelftoets.md        ← check of je het echt snapt
```

## Hoe werk je een module door?

1. Lees de **module-README** voor de leerdoelen en het leespad.
2. Wissel **notes** af met **video's**. De notes verwijzen naar de juiste momenten.
3. Doe de **opdrachten**. De code-opdrachten hebben ingebouwde tests: als `python oX_....py` "Alle tests geslaagd!" print, ben je klaar.
4. Maak de **zelftoets** zonder te spieken. Kun je een vraag niet beantwoorden, ga dan terug naar de notes.
