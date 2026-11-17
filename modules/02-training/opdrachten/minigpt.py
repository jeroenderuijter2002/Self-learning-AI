"""
minigpt.py: een compacte GPT voor de scaling-experimenten in module 2
=====================================================================

Dit is dezelfde architectuur die je in module 1 (opdracht O3) zelf hebt gebouwd,
maar dan in één bestand van ~120 regels, zodat je hem makkelijk in verschillende
groottes kunt trainen. Heb je je eigen O3-code nog? Gebruik die gerust; het enige
wat O1 nodig heeft is een model dat je in een paar groottes kunt maken.

    from minigpt import GPT, Config, load_shakespeare, train

Karakter-niveau, tiny Shakespeare (~1,1 miljoen tekens), pre-norm blocks,
geleerde positie-embeddings, weight tying. Draait op een laptop-CPU.
"""
import math, time, urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd))
        self.n_head = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying (zie 1.1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def n_params(self, non_embedding: bool = True) -> int:
        """Aantal parameters. Kaplan et al. tellen zonder embeddings; dat doen wij ook."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok.weight.numel() + self.pos.weight.numel()
        return n

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.c.block_size:])
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx


def load_shakespeare(path="tinyshakespeare.txt"):
    """Geeft (train_ids, val_ids, encode, decode). 90/10-split, karakter-niveau."""
    p = Path(path)
    if not p.exists():
        urllib.request.urlretrieve(URL, p)
    text = p.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], (lambda s: [stoi[c] for c in s]), (lambda ids: "".join(chars[i] for i in ids))


def get_batch(data, block_size, batch_size, gen=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def lr_at(step, max_steps, lr_max, warmup):
    """Lineaire warmup, daarna cosine decay naar 10% van lr_max (zie notes 2.1)."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return lr_max * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def estimate_val_loss(model, val, block_size, n_batches=20, batch_size=64, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)          # elke keer dezelfde val-batches
    losses = [model(*get_batch(val, block_size, batch_size, g))[1].item() for _ in range(n_batches)]
    model.train()
    return sum(losses) / len(losses)


def train(cfg: Config, train_data, val_data, max_steps=1500, batch_size=32,
          lr=1.5e-3, warmup=100, eval_every=100, seed=0, verbose=True):
    """Train één model. Geeft het model en een log met (stap, tokens, train-loss, val-loss, lr)."""
    torch.manual_seed(seed)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    log, t0, g = [], time.time(), torch.Generator().manual_seed(seed)
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps:
            v = estimate_val_loss(model, val_data, cfg.block_size)
            log.append({"step": step, "tokens": step * batch_size * cfg.block_size,
                        "val": v, "lr": lr_at(step, max_steps, lr, warmup)})
            if verbose:
                print(f"  stap {step:5d} | val {v:.3f} | {time.time() - t0:5.0f}s", flush=True)
        if step == max_steps:
            break
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, max_steps, lr, warmup)
        x, y = get_batch(train_data, cfg.block_size, batch_size, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and "train" not in log[-1]:
            log[-1]["train"] = loss.item()
    return model, log


if __name__ == "__main__":
    tr, va, enc, dec = load_shakespeare()
    model, log = train(Config(n_layer=2, n_embd=64, n_head=4), tr, va, max_steps=300)
    print(f"{model.n_params():,} parameters (zonder embeddings)")
    print(dec(model.generate(torch.zeros((1, 1), dtype=torch.long), 200)[0].tolist()))
