# Opdracht O1: Bouw je eigen BPE-tokenizer

*Module 1* · ⏱️ 1–2 uur · 📖 eerst: [1.1 Tokens en embeddings](../notes/1.1-tokens-en-embeddings.md)

## Doel
Het tokenization-algoritme achter GPT en Claude helemaal begrijpen, door het zelf te implementeren in ~50 regels Python.

## Deel A: implementeren (verplicht)

Open [`o1_bpe.py`](o1_bpe.py) en implementeer:

| Functie | Wat |
|---------|-----|
| `get_pair_counts(ids)` | tel opeenvolgende paren |
| `merge(ids, pair, new_id)` | vervang een paar door een nieuw token |
| `BPETokenizer.train(text, vocab_size)` | leer merges |
| `BPETokenizer.encode(text)` | tekst → ids |
| `BPETokenizer.decode(ids)` | ids → tekst |

```bash
python o1_bpe.py
# Compressie: 1980 bytes -> ... tokens (...x)
# Alle tests geslaagd!
```

<details>
<summary>💡 Hint voor <code>encode</code></summary>

Je kunt merges niet in willekeurige volgorde toepassen. Token 300 kan bestaan uit token 280 + token 101, dus 280 moet eerst bestaan. Zoek in elke iteratie het paar in je huidige `ids` met de **laagste** waarde in `self.merges`, en merge dat. Stop als geen enkel paar nog in `self.merges` zit.
</details>

<details>
<summary>💡 Hint voor <code>decode</code></summary>

Houd in `self.vocab` bij welke **bytes** elk token voorstelt. Een nieuw token is `vocab[a] + vocab[b]`. Decoderen is dan: bytes aan elkaar plakken en `.decode("utf-8")`. Wat doe je als de bytes geen geldige UTF-8 zijn? (Kijk naar de `errors`-parameter.)
</details>

## Deel B: experimenten (aanbevolen)

1. **Train op echte tekst.** Download [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) en train met `vocab_size=512`, met `verbose=True`. Welke merges komen eerst? Herken je woorden?
2. **Nederlands tegenover Engels.** Train je tokenizer op Engelse tekst en encodeer dan een Nederlandse alinea (bijvoorbeeld een stukje van je thesis). Vergelijk de compressieratio. Wat zegt dit over hoe "duur" Nederlands is voor een model met een Engels-zware tokenizer?
3. **Vergelijk met het echte werk:**
   ```python
   import tiktoken
   enc = tiktoken.get_encoding("cl100k_base")   # de GPT-4-tokenizer
   print(enc.encode("Hallo, ik ben Jeroen en ik studeer in Eindhoven."))
   print([enc.decode([t]) for t in enc.encode("ongezelligheidsfactor")])
   ```
   Hoe knipt een tokenizer met 100k tokens een lang Nederlands woord op?

## Deel C: uitbreiding (optioneel, voor als je het leuk vindt)

Echte tokenizers splitsen tekst eerst met een **regex**, zodat merges nooit over woordgrenzen of tussen letters en leestekens heen gaan. Anders krijg je tokens als `"dog."` en `"dog!"` als aparte entries. Implementeer een `RegexTokenizer` met het GPT-4-patroon. Karpathy's [minbpe](https://github.com/karpathy/minbpe) heeft hiervoor een stapsgewijze [`exercise.md`](https://github.com/karpathy/minbpe/blob/master/exercise.md).

🎬 **Wil je meer diepgang?** Karpathy's [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (2u13m) bouwt precies dit en laat zien welke rare LLM-bugs eruit voortkomen.

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je in eigen woorden kunt uitleggen waarom `"strawberry"` letters tellen lastig maakt voor een LLM.

**Oplossing:** [`../oplossingen/o1_bpe_oplossing.py`](../oplossingen/o1_bpe_oplossing.py)
