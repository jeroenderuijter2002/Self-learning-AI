# Opdracht O3: Bouw je eigen GPT 🏆

*Module 1* · ⏱️ 4–6 uur (de hoofdopdracht) · 📖 eerst: [1.3 Het transformer-block](../notes/1.3-transformer-block.md)

## Doel
Een complete GPT vanaf een leeg bestand bouwen en trainen, die Shakespeare-achtige tekst genereert. Na deze opdracht heb je elke regel code van een taalmodel zelf geschreven.

## Aanpak: code-along met Karpathy

🎬 **[Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1u56m)

Dit is waarschijnlijk de beste tutorial over transformers die er bestaat. **Kijk niet passief: typ alles zelf mee.** Pauzeer, probeer de volgende stap eerst zelf, en kijk dan hoe hij het doet.

- Bijbehorende code: [github.com/karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
- Data: [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1 MB)
- **Geen GPU?** Het eindmodel traint op een laptop-CPU in 15–30 minuten als je het verkleint (zie tabel). Op Colab met een T4-GPU kan het volledige model.

### Mijlpalen in de video

Vink af terwijl je bouwt. Bij elke mijlpaal moet je code draaien.

| ✓ | Mijlpaal | Wat je nu hebt | Verwachte val-loss |
|---|----------|----------------|--------------------|
| ☐ | Data laden, character-level tokenizer | `encode`/`decode` | n.v.t. |
| ☐ | Bigram-model | baseline, genereert onzin | ~2,5 |
| ☐ | "Wiskundige truc": averaging met `tril` | het masker uit O2, in PyTorch | n.v.t. |
| ☐ | Eén self-attention head | eerste echte attention | ~2,4 |
| ☐ | Multi-head attention | | ~2,3 |
| ☐ | + Feed-forward (MLP) | | ~2,2 |
| ☐ | + Residuals + LayerNorm, gestapelde blocks | volledige transformer | ~2,0 |
| ☐ | + Dropout, opschalen | een echte mini-GPT | ~1,5 (GPU) |

### Kleine config voor CPU

| Hyperparameter | Karpathy (GPU) | CPU-vriendelijk |
|----------------|----------------|-----------------|
| `n_embd` | 384 | 128 |
| `n_head` | 6 | 4 |
| `n_layer` | 6 | 4 |
| `block_size` | 256 | 64 |
| `batch_size` | 64 | 32 |
| `max_iters` | 5000 | 3000 |

Op Apple Silicon: `device = "mps"`.

## Daarna: maak het je eigen (kies er minstens 2)

Hier leer je het meest. Pas de code aan en **meet het effect** op de validatie-loss.

1. **🔤 Je eigen tokenizer erin.** Vervang de character-level tokenizer door je BPE uit **O1** (`vocab_size` ≈ 512). Wordt de tekst beter? Let op: de loss per token is niet meer vergelijkbaar met per karakter. Waarom niet, en hoe corrigeer je dat? (Hint: bits per character.)
2. **🇳🇱 Nederlands model.** Train op een Nederlandse tekst, bijvoorbeeld een boek van [Project Gutenberg](https://www.gutenberg.org/browse/languages/nl) zoals *Max Havelaar* of *De Camera Obscura*. Laat je model Multatuli nadoen.
3. **🔬 Ablation study.** Haal één onderdeel weg en meet: geen residuals, geen LayerNorm, geen MLP of geen positie-embedding. Welke ingreep doet het meeste pijn? Maak een tabelletje. Dit soort experimenten doen onderzoekers echt.
4. **🌡️ Sampling.** Implementeer temperatuur en top-k in `generate()`. Genereer met $\tau$ = 0,5 / 1,0 / 1,5. Beschrijf het verschil.
5. **👀 Kijk erin.** Sla de attention-gewichten van een getraind model op en plot ze voor een zin. Vind je heads met een duidelijk patroon?
6. **🌀 RoPE.** Vervang de geleerde positie-embedding door RoPE (zie [1.4](../notes/1.4-van-model-naar-tekst.md)). Pittig, maar heel leerzaam.

## Nog verder? (na deze module)
- 💻 **[nanoGPT](https://github.com/karpathy/nanoGPT)**: dezelfde code, maar productierijp. Kan GPT-2 reproduceren.
- 🎬 **[*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU)** (4 uur): van dit speelgoedmodel naar een echte GPT-2, inclusief GPU-optimalisaties.
- 💻 **[nanochat](https://github.com/karpathy/nanochat)**: een complete ChatGPT-kloon (pretraining, fine-tuning, chat-UI) voor ~$100 aan GPU-tijd. Een perfecte brug naar module 2.

## ✅ Je bent klaar als
- [ ] je model leesbare(-achtige) Shakespeare genereert;
- [ ] je minstens 2 uitbreidingen hebt gedaan, met gemeten resultaten;
- [ ] je je eigen `model.py` regel voor regel aan iemand kunt uitleggen.

💡 **Tip:** zet je O3-code op je eigen GitHub. Een zelfgebouwde GPT met ablation study is een sterk portfolio-item voor sollicitaties.
