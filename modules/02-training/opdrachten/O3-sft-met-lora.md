# Opdracht O3: Supervised fine-tuning met je eigen LoRA

*Module 2* · ⏱️ 2–4 uur · 📖 eerst: [2.3 Supervised fine-tuning](../notes/2.3-supervised-fine-tuning.md)

## Doel

Een basismodel dat alleen documenten afmaakt, veranderen in een (kleine) assistent. Je schrijft de twee onderdelen die het verschil maken zelf: een **LoRA-laag** en het **loss-masker** voor een chat-template. Daarna finetune je SmolLM2-135M, het basismodel, en meet je vóór en na. Dit is een voorproefje van module 4 (LoRA, QLoRA) en precies wat je nodig hebt voor de praktijkopdracht.

## Deel A: de bouwstenen (verplicht)

Open [`o3_lora_sft.py`](o3_lora_sft.py) en implementeer:

| Functie / klasse | Wat |
|---|---|
| `LoRALinear` | een bevroren `nn.Linear` plus $\frac{\alpha}{r} B A$, met $B = 0$ bij de start |
| `LoRALinear.merged()` | de LoRA-correctie in de gewichten "inbakken" |
| `add_lora(model, targets)` | alle `q_proj`, `k_proj`, `v_proj`, `o_proj` vervangen, de rest bevriezen |
| `build_sft_example(tok, messages)` | `input_ids` plus `labels` met −100 op alles vóór het antwoord |

```bash
python o3_lora_sft.py
# Gesprek: 45 tokens, waarvan 7 met loss
# Alle tests geslaagd!
```

De tests controleren dat je LoRA-laag bij de start **precies** de originele laag is, dat alleen $A$ en $B$ trainbaar zijn, dat `merged()` dezelfde uitvoer geeft, en dat in je labels alleen het antwoord **plus het `<|im_end|>`-token** meetelt. Dat laatste is de reden dat het model na SFT weet wanneer het moet stoppen.

<details>
<summary>💡 Hint: waarom is B nul en A niet?</summary>

Als allebei nul zijn, is de gradiënt naar $A$ en $B$ ook nul ($\partial/\partial A$ bevat $B$ en omgekeerd), en leert er niets. Met $B = 0$ en een willekeurige $A$ begint het model als het origineel, maar krijgt $B$ meteen een gradiënt.
</details>

<details>
<summary>💡 Hint: modules vervangen</summary>

```python
for name, module in list(model.named_modules()):
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name.endswith(targets):
            setattr(module, child_name, LoRALinear(child, r, alpha))
```
Bevries eerst alles (`p.requires_grad = False`), en vervang daarna: de nieuwe $A$ en $B$ zijn dan als enige trainbaar.
</details>

## Deel B: finetunen en meten (verplicht)

```bash
python o3_lora_sft.py --train
```

Het script doet het volgende:
1. Het laadt het **basismodel** SmolLM2-135M (niet de Instruct-versie) en 450 instructies uit [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).
2. Het meet de loss op 50 achtergehouden antwoorden, en laat het model drie vragen beantwoorden in de chat-template.
3. Het voegt jouw LoRA toe ($r = 16$) en traint 150 stappen van 4 gesprekken.
4. Het meet opnieuw.

Op een laptop-CPU duurt dit ongeveer 10–20 minuten; op Colab met een T4-GPU een paar minuten (zet dan `model.to("cuda")` en de batches op dezelfde device).

Zo zag het eruit toen ik de oplossing draaide (CPU, 1 thread, ~25 min):

| | Basismodel | Na 150 stappen LoRA-SFT |
|---|---|---|
| val-loss op de antwoorden | 2,00 | 1,64 |
| stopt netjes met `<|im_end|>` | 0 van 3 | 3 van 3 |

Vóór de training maakte het basismodel van *"Explain what a black hole is in two sentences."* een willekeurig webdocument: een nieuwe vraag, gevolgd door een verhaal over de NSA. Erna: *"A black hole is a region of space where gravity is so strong that nothing, not even light, can escape its gravitational pull."*, en dan stopt het.

> 🐛 **Een echte bug, en wat hij leert.** De eerste keer dat ik dit draaide, leerde het model wel het format, maar stopte het nóóit. Op de plek van `<|im_end|>` verschenen rare tokens als "pvproperty" en "LEGATO". De oorzaak: het basismodel heeft `<|im_end|>` in pretraining nooit gezien, dus de embedding van dat token is ongetraind, en lijkt (cosine-similariteit 0,99) op die van andere ongetrainde tokens zoals "pvproperty". Omdat SmolLM2 weight tying gebruikt (dezelfde matrix voor in- en uitvoer, 🔁 notes 1.1), kan LoRA op de attention-lagen het model nooit leren om juist dát token te kiezen. De oplossing staat in `unfreeze_token_rows`: train de twee embedding-rijen van de speciale tokens mee (1.152 extra parameters). In `peft` heet dit `modules_to_save` of `trainable_token_indices`. Onthoud dit als je ooit speciale tokens toevoegt aan een basismodel.

**Beantwoord:**
1. Hoeveel parameters train je, als percentage van het hele model? Hoe groot zou het LoRA-bestand zijn in float16?
2. Wat gebeurt er met de val-loss op de antwoorden? En met het aantal antwoorden dat netjes stopt met `<|im_end|>`?
3. Lees de antwoorden vóór en na. Wat is er veranderd: de **kennis** of de **vorm**? Verbind dit met de Superficial Alignment Hypothesis en figuur 2.8.
4. Vergelijk je eigen model met SmolLM2-135M-**Instruct** (verander `BASE_MODEL` tijdelijk). Waar zit het verschil nog? (Hint: Hugging Face deed SFT op een grote synthetische dataset, [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), en daarna DPO.)

## Deel C: maak het je eigen (kies er minstens één)

1. **Een eigen stijl.** Schrijf (of laat een groot model schrijven) 100–200 gesprekken in een duidelijk herkenbare stijl: altijd in het Nederlands, altijd eindigend met één concrete vervolgactie, of altijd in de toon van een vriendelijke huisarts. Finetune daarop. Hoeveel voorbeelden zijn er nodig voordat het model de stijl oppikt?
2. **Rang en targets.** Vergelijk $r = 2, 8, 32$, en LoRA op alleen attention versus ook op de MLP (`gate_proj`, `up_proj`, `down_proj`). Meet de val-loss. Wat levert meer op?
3. **Het masker uitzetten.** Train een keer zonder loss-masker (labels = input_ids). Wat gebeurt er met de val-loss op de antwoorden, en met het gedrag? Kun je verklaren waarom?
4. **Tegen peft.** Vervang jouw `add_lora` door `peft.get_peft_model` met dezelfde $r$, $\alpha$ en targets. Krijg je (op toeval na) dezelfde resultaten?
5. **De bug zelf zien.** Draai `run_training(train_special=False)`. Stopt het model nog? Print voor een antwoord de top-5 tokens op de plek waar `<|im_end|>` had moeten komen. Herken je de verklaring hierboven?

## ✅ Je bent klaar als
- [ ] alle tests slagen;
- [ ] je een vóór/na-vergelijking hebt met getallen (val-loss, stopgedrag) én voorbeelden;
- [ ] je in eigen woorden kunt uitleggen wat LoRA doet, waarom $B = 0$ begint, en waarom het loss-masker ertoe doet.

**Oplossing:** [`../oplossingen/o3_lora_sft_oplossing.py`](../oplossingen/o3_lora_sft_oplossing.py)
