"""
Oplossing O3: Supervised fine-tuning met je eigen LoRA
======================================================
(Kijk pas als je vastzit.)

Deel A: vul de TODO's in en draai de tests (downloadt alleen de tokenizer, ~2 MB):

    python o3_lora_sft.py

Deel B: finetune het BASISmodel SmolLM2-135M tot een (kleine) assistent en vergelijk
vóór en na. Op een laptop-CPU ~10-20 min, op Colab (T4) een paar minuten:

    python o3_lora_sft.py --train
"""
import math
import sys

import torch
import torch.nn as nn

TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"   # zelfde vocabulaire als het basismodel, mét chat-template
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"


class LoRALinear(nn.Module):
    """Een bevroren nn.Linear plus een trainbare correctie van lage rang (notes 2.3):

        y = base(x) + (alpha / r) * x @ A^T @ B^T

    A: (r, in_features), init met kleine random waarden (bijv. nn.init.kaiming_uniform_(A, a=math.sqrt(5)))
    B: (out_features, r), init op NUL  -> bij de start is de laag precies de originele laag.
    Zet requires_grad=False op alle parameters van `base`.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Een gewone nn.Linear met W' = W + scale * B @ A (handig om zonder overhead te deployen)."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def add_lora(model: nn.Module, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=8, alpha=16.0) -> nn.Module:
    """Vervang elke nn.Linear waarvan de naam eindigt op een van `targets` door een LoRALinear.
    Bevries daarna ALLE andere parameters van het model. Geeft het model terug.

    Hint: loop over list(model.named_modules()), en gebruik setattr(parent, kindnaam, nieuwe_laag).
    """
    for p in model.parameters():
        p.requires_grad = False
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and child_name.endswith(tuple(targets)):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_sft_example(tokenizer, messages, max_len: int = 384):
    """Maak (input_ids, labels) voor één gesprek, met een loss-masker (notes 2.3).

    messages: lijst van {"role": ..., "content": ...}; het LAATSTE bericht is het antwoord van de assistent.
    - input_ids: de token-ids van het volledige gesprek volgens tokenizer.apply_chat_template.
    - labels:    gelijk aan input_ids, maar -100 op alle posities vóór het antwoord
                 (PyTorch's cross_entropy negeert -100). Het antwoord én het afsluitende
                 <|im_end|> moeten wél meetellen: zo leert het model stoppen.
    Kap af op max_len tokens.

    Hint: apply_chat_template(messages[:-1], add_generation_prompt=True) geeft precies het deel vóór het antwoord.
    Let op: afhankelijk van je transformers-versie geeft apply_chat_template(..., tokenize=True) een lijst
    of een dict-achtig object terug; in het tweede geval staan de ids onder ["input_ids"].
    Returns: (input_ids: list[int], labels: list[int])
    """
    def ids_of(x):
        return list(x["input_ids"]) if hasattr(x, "keys") else list(x)
    full = ids_of(tokenizer.apply_chat_template(messages, tokenize=True))
    prompt = ids_of(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
    assert full[:len(prompt)] == prompt, "de prompt moet een prefix van het gesprek zijn"
    labels = [-100] * len(prompt) + full[len(prompt):]
    return full[:max_len], labels[:max_len]


def sft_loss(logits, labels):
    """Gemiddelde cross-entropy over de posities met label != -100, met de verschuiving t -> t+1."""
    return nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)


def unfreeze_token_rows(model, token_ids):
    """Maak alleen de embedding-rijen van `token_ids` trainbaar (gegeven, geen TODO).

    Waarom? Het basismodel heeft <|im_start|> en <|im_end|> in pretraining (bijna) nooit gezien.
    Hun embeddings zijn dus ongetraind, en lijken sterk op die van andere ongetrainde tokens
    (bij SmolLM2-135M heeft <|im_end|> een cosine-similariteit van 0,99 met het token 'pvproperty').
    Omdat de uitvoerlaag hier dezelfde matrix is als de embedding (weight tying, notes 1.1),
    kan LoRA op de attention-lagen het model nooit leren om juist <|im_end|> te kiezen.
    We trainen daarom die paar rijen mee; de gradiënt van alle andere rijen zetten we op nul.
    """
    emb = model.get_input_embeddings().weight
    mask = torch.zeros(emb.shape[0], 1, dtype=emb.dtype)
    mask[list(token_ids)] = 1.0
    emb.requires_grad_(True)
    emb.register_hook(lambda g: g * mask)
    return len(token_ids) * emb.shape[1]


# ---------------------------------------------------------------------------
# Deel B: echt finetunen
# ---------------------------------------------------------------------------
EVAL_PROMPTS = ["What are three tips for staying focused while studying?",
                "Explain what a black hole is in two sentences.",
                "Write a short, polite email declining a meeting invitation."]


def _collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100); att = torch.zeros(len(batch), L, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids, lab, att


@torch.no_grad()
def evaluate(model, tok, val, bs=2):
    model.eval(); tot = 0.0
    for i in range(0, len(val), bs):
        ids, lab, att = _collate(val[i:i + bs], tok.pad_token_id)
        tot += sft_loss(model(input_ids=ids, attention_mask=att).logits, lab).item() * len(val[i:i + bs])
    model.train()
    return tot / len(val)


@torch.no_grad()
def show(model, tok, title):
    model.eval()
    print(f"\n----- {title} -----")
    stops = 0
    for q in EVAL_PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"]
        # Let op: geen repetition_penalty. Die straft élk token dat al in de context staat, ook
        # <|im_end|> (dat in de prompt voorkomt), waardoor het model niet meer wil stoppen.
        out = model.generate(ids, max_new_tokens=100, do_sample=False, no_repeat_ngram_size=4,
                             eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
        new = out[0, ids.shape[1]:]
        stops += int(tok.convert_tokens_to_ids("<|im_end|>") in new.tolist())
        print(f"\n> {q}\n{tok.decode(new, skip_special_tokens=True).strip()[:400]}")
    print(f"\n[stopt netjes met <|im_end|>: {stops}/{len(EVAL_PROMPTS)}]")
    model.train()


def run_training(n_train=400, n_val=50, steps=150, bs=4, lr=3e-4, r=16, train_special=True):
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = [x for x in ds.select(range(5000)) if not x["input"] and len(x["output"]) < 700]
    random.Random(0).shuffle(rows)
    conv = lambda x: [{"role": "user", "content": x["instruction"]}, {"role": "assistant", "content": x["output"]}]
    data = [build_sft_example(tok, conv(x)) for x in rows[:n_train + n_val]]
    train, val = data[:n_train], data[n_train:]
    print(f"Basismodel: val-loss op de antwoorden = {evaluate(model, tok, val):.3f}")
    show(model, tok, "VÓÓR fine-tuning (basismodel met chat-template)")
    model = add_lora(model, r=r, alpha=2 * r)
    n_lora = n_trainable(model)
    n_rows = unfreeze_token_rows(model, tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])) if train_special else 0
    print(f"\nTrainbare parameters: {n_lora:,} (LoRA) + {n_rows:,} (2 embedding-rijen) "
          f"van {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    for step in range(steps):
        ids, lab, att = _collate(random.Random(step).sample(train, bs), tok.pad_token_id)
        loss = sft_loss(model(input_ids=ids, attention_mask=att).logits, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"stap {step:4d} | train-loss {loss.item():.3f}", flush=True)
    print(f"\nNa SFT: val-loss op de antwoorden = {evaluate(model, tok, val):.3f}")
    show(model, tok, "NA fine-tuning (LoRA)")


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
def _tests():
    torch.manual_seed(0)
    base = nn.Linear(32, 48)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 32)
    assert torch.allclose(lora(x), base(x)), "bij de start moet LoRA precies de originele laag zijn (B = 0)"
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["A", "B"], f"alleen A en B mogen trainbaar zijn, niet {trainable}"
    assert lora.A.shape == (4, 32) and lora.B.shape == (48, 4)
    with torch.no_grad():
        lora.B.normal_()
    expected = base(x) + 2.0 * x @ lora.A.T @ lora.B.T
    assert torch.allclose(lora(x), expected, atol=1e-5), "y = base(x) + (alpha/r) * x A^T B^T"
    assert torch.allclose(lora.merged()(x), lora(x), atol=1e-5), "merged() moet dezelfde uitvoer geven"
    lora(x).sum().backward()
    assert base.weight.grad is None and lora.A.grad is not None

    # add_lora op een mini-'transformer'
    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.q_proj, s.k_proj, s.v_proj, s.o_proj = (nn.Linear(16, 16) for _ in range(4))
    class Toy(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(10, 16)
            s.layers = nn.ModuleList([nn.ModuleDict({"self_attn": Attn(), "mlp": nn.Linear(16, 16)}) for _ in range(2)])
    toy = add_lora(Toy(), r=2, alpha=4)
    assert isinstance(toy.layers[0]["self_attn"].q_proj, LoRALinear)
    assert isinstance(toy.layers[1]["mlp"], nn.Linear) and not isinstance(toy.layers[1]["mlp"], LoRALinear)
    assert n_trainable(toy) == 2 * 4 * (2 * 16 + 16 * 2), f"verwacht 512 trainbare parameters, niet {n_trainable(toy)}"

    # loss-masker met de echte SmolLM2-tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    msgs = [{"role": "user", "content": "Wat is 2+2?"}, {"role": "assistant", "content": "Dat is 4."}]
    ids, labels = build_sft_example(tok, msgs)
    full = tok.apply_chat_template(msgs, tokenize=True)
    full = full["input_ids"] if hasattr(full, "keys") else list(full)
    assert ids == full[:len(ids)], "input_ids moeten het volledige gesprek zijn"
    assert len(ids) == len(labels)
    kept = [t for t in labels if t != -100]
    assert tok.decode(kept).startswith("Dat is 4.<|im_end|>"), f"alleen het antwoord + <|im_end|> telt mee: {tok.decode(kept)!r}"
    n_prompt = labels.index(next(t for t in labels if t != -100))
    assert all(t == -100 for t in labels[:n_prompt])
    print(f"Gesprek: {len(ids)} tokens, waarvan {len(kept)} met loss")
    print("Alle tests geslaagd!")


if __name__ == "__main__":
    if "--train" in sys.argv:
        run_training()
    else:
        _tests()
