"""
Experimenten voor module 2 (Van tekstvoorspeller naar assistent).

Elk experiment schrijft een klein JSON-bestand naar tools/results/m2_<naam>.json.
tools/make_figures_module02.py maakt daar de figuren van. De resultaten staan in de
repo, zodat je de figuren kunt hermaken zonder alles opnieuw te draaien.

    pip install numpy torch transformers datasets matplotlib scipy
    python tools/m2_experiments.py scaling     # 6 mini-GPT's op tiny Shakespeare   (~20 min CPU)
    python tools/m2_experiments.py lr          # warmup/cosine vs te hoge lr        (~5 min)
    python tools/m2_experiments.py pythia      # optellen bij Pythia 70M-1.4B            (~30 min)
    python tools/m2_experiments.py digits      # 'mirage': Qwen2.5 0.5B/1.5B, 1-8 cijfers (~25 min)
    python tools/m2_experiments.py sftdiff     # Qwen2.5-0.5B basis vs Instruct      (~5 min)
    python tools/m2_experiments.py rm          # reward model op HH-RLHF             (~25 min)
    python tools/m2_experiments.py dpo         # kleine DPO-run (SmolLM2-135M+LoRA)  (~20 min)
    python tools/m2_experiments.py gsm8k       # majority voting, Qwen2.5-0.5B-Instr (~45 min)

Alles draait op een CPU; tijden gemeten op 2 cores.
"""
import json, math, random, re, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
RES = Path(__file__).resolve().parent / "results"
DATA.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "modules/02-training/opdrachten"))
torch.set_num_threads(max(1, torch.get_num_threads()))


def dump(name, obj):
    p = RES / f"m2_{name}.json"
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print("geschreven:", p)


# --------------------------------------------------------------------------- scaling
SIZES = [(1, 32), (2, 64), (3, 96), (4, 128), (4, 192), (6, 256)]   # (n_layer, n_embd)


def exp_scaling():
    from minigpt import Config, load_shakespeare, train
    tr, va, _, _ = load_shakespeare(DATA / "tinyshakespeare.txt")
    runs = []
    for L, d in SIZES:
        cfg = Config(n_layer=L, n_embd=d, n_head=4, block_size=64)
        print(f"model L={L} d={d}", flush=True)
        t0 = time.time()
        model, log = train(cfg, tr, va, max_steps=2000, batch_size=32, lr=1.5e-3, warmup=100, eval_every=100)
        runs.append({"n_layer": L, "n_embd": d, "N": model.n_params(), "N_total": model.n_params(False),
                     "seconds": time.time() - t0, "log": log})
        dump("scaling", {"block_size": 64, "batch_size": 32, "runs": runs})


def exp_lr():
    from minigpt import Config, load_shakespeare, train
    tr, va, _, _ = load_shakespeare(DATA / "tinyshakespeare.txt")
    cfg = Config(n_layer=4, n_embd=128, n_head=4, block_size=64)
    out = []
    for name, lr, warm in [("warmup + cosine, lr 2e-3", 2e-3, 100),
                           ("geen warmup, lr 2e-2", 2e-2, 1),
                           ("warmup + cosine, lr 2e-2", 2e-2, 100)]:
        print(name, flush=True)
        _, log = train(cfg, tr, va, max_steps=1000, batch_size=32, lr=lr, warmup=warm, eval_every=10, verbose=False)
        out.append({"name": name, "lr": lr, "warmup": warm, "log": log})
        dump("lr", out)


# --------------------------------------------------------------------------- emergentie (Pythia)
PYTHIA = ["70m", "160m", "410m", "1b", "1.4b"]   # 2.8b past niet in 8 GB RAM


def _addition_problems(n, digits, seed):
    rng = random.Random(seed)
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]


def _spaced(n):
    return " ".join(str(n))


def exp_pythia():
    """Optellen bij steeds grotere Pythia-modellen (zelfde data, zelfde volgorde).

    De cijfers staan met spaties ertussen ("4 7 + 3 8 = 8 5"), zodat elk cijfer precies één
    token is. Per model drie meetlatten op dezelfde 150 sommen:
      - exact match: greedy genereren, alle cijfers goed;
      - cijfers goed: teacher-forced, het aandeel antwoordcijfers dat het model als top-1 voorspelt;
      - kans op het juiste antwoord: het product van de cijferkansen.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = RES / "m2_pythia.json"
    out = json.loads(path.read_text()) if path.exists() else {"models": []}   # hervat waar je was
    done = {m["size"] for m in out["models"]}
    for size in PYTHIA:
        if size in done:
            continue
        name = f"EleutherAI/pythia-{size}"
        print(name, flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        big = size in ("1.4b", "2.8b")          # bf16 om binnen ~8 GB RAM te blijven
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if big else torch.float32).eval()
        row = {"name": name, "size": size, "N": sum(p.numel() for p in model.parameters()), "tasks": {}}
        for digits in (2, 3):
            fs = "".join(f"{_spaced(a)} + {_spaced(b)} = {_spaced(a + b)}\n" for a, b in _addition_problems(8, digits, seed=1))
            em, dig, probs = [], [], []
            for a, b in _addition_problems(150, digits, seed=2):
                p_ids = tok(fs + f"{_spaced(a)} + {_spaced(b)} =").input_ids
                a_ids = tok(" " + _spaced(a + b)).input_ids
                with torch.no_grad():
                    lp = torch.log_softmax(model(torch.tensor([p_ids + a_ids])).logits[0].float(), -1)
                pos = range(len(p_ids) - 1, len(p_ids) + len(a_ids) - 1)
                dig.append(np.mean([lp[i].argmax().item() == t for i, t in zip(pos, a_ids)]))
                probs.append(math.exp(sum(lp[i, t].item() for i, t in zip(pos, a_ids))))
                with torch.no_grad():
                    g = model.generate(torch.tensor([p_ids]), max_new_tokens=len(a_ids) + 1, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                txt = tok.decode(g[0, len(p_ids):]).split("\n")[0].replace(" ", "")
                em.append(txt == str(a + b))
            row["tasks"][str(digits)] = {"exact_match": float(np.mean(em)), "digit_acc": float(np.mean(dig)),
                                         "p_correct": float(np.mean(probs)), "n": len(em)}
            print("  ", digits, row["tasks"][str(digits)], flush=True)
        out["models"].append(row)
        out["models"].sort(key=lambda m: m["N"])
        dump("pythia", out)
        del model
        import gc; gc.collect()


def exp_digits(models=("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"), max_digits=8, n=100):
    """Het 'mirage'-mechanisme van Schaeffer et al.: optellen met 1 tot 8 cijfers.
    Qwen knipt getallen in losse cijfers, dus elk cijfer is één token. Per aantal cijfers k:
    exact match (greedy, hele getal goed) en de fractie cijfers die teacher-forced goed is."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    out = {"models": []}
    for name in models:
        print(name, flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
        row = {"name": name, "N": sum(p.numel() for p in model.parameters()), "k": {}}
        for k in range(1, max_digits + 1):
            fs = "".join(f"{a} + {b} = {a + b}\n" for a, b in _addition_problems(8, k, seed=1))
            em, dig = [], []
            for a, b in _addition_problems(n, k, seed=2):
                p_ids = tok(fs + f"{a} + {b} =").input_ids
                a_ids = tok(f" {a + b}").input_ids
                with torch.no_grad():
                    lp = model(torch.tensor([p_ids + a_ids])).logits[0].float()
                    g = model.generate(torch.tensor([p_ids]), max_new_tokens=len(a_ids) + 1, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                pos = range(len(p_ids) - 1, len(p_ids) + len(a_ids) - 1)
                digit_pos = [(i, t) for i, t in zip(pos, a_ids) if tok.decode([t]).strip().isdigit()]
                dig.append(np.mean([lp[i].argmax().item() == t for i, t in digit_pos]))
                em.append(tok.decode(g[0, len(p_ids):]).split("\n")[0].strip() == str(a + b))
            row["k"][str(k)] = {"exact_match": float(np.mean(em)), "digit_acc": float(np.mean(dig)), "n": n}
            print("  ", k, row["k"][str(k)], flush=True)
        out["models"].append(row)
        dump("digits", out)
        del model


# --------------------------------------------------------------------------- basis vs instruct
CHAT_PROMPTS = [
    "What is the capital of France?",
    "Explain in two sentences what a neural network is.",
    "Give me three tips for a job interview.",
    "Write a short poem about the sea.",
    "Translate into Dutch: The weather is nice today.",
    "Wat is de hoofdstad van Nederland?",
    "Write a Python function that checks whether a number is prime.",
    "Can you help me write an excuse for missing work tomorrow?",
]

URIAL = """# Instructions

Below are conversations between a curious human and a helpful, honest AI assistant. The assistant answers the question directly and clearly, and then stops.

# Query:
```What is the boiling point of water?```

# Answer:
```
At sea level, water boils at 100 °C (212 °F). At higher altitudes the boiling point is lower, because the air pressure is lower.
```

# Query:
```Give me two ideas for a rainy afternoon.```

# Answer:
```
1. Bake something new, for example a simple banana bread.
2. Start a jigsaw puzzle or a board game with friends.
```

# Query:
```{q}```

# Answer:
```
"""


def _qwen(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval()


def _gen(tok, model, text, n, temperature, max_new=110, seed=0, stop=None):
    ids = tok(text, return_tensors="pt").input_ids
    outs = []
    for k in range(n):
        torch.manual_seed(seed + k)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=max_new, do_sample=temperature > 0,
                               temperature=temperature if temperature > 0 else None, top_p=0.95 if temperature > 0 else None,
                               top_k=None, pad_token_id=tok.eos_token_id)
        s = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
        for st in (stop or []):
            if st in s:
                s = s[:s.index(st)]
        outs.append(s.replace("<|endoftext|>", "").replace("<|im_end|>", "").rstrip())
    return outs


def exp_sftdiff():
    """Qwen2.5-0.5B (basis) tegenover Qwen2.5-0.5B-Instruct.
    1) Completions voor de speeltuin: basis met kale prompt, basis met URIAL-prompt, instruct.
    2) Token-analyse (zoals Lin et al. 2023): voor elk token van het instruct-antwoord,
       welke rang heeft dat token onder het basismodel, met dezelfde context?"""
    btok, base = _qwen("Qwen/Qwen2.5-0.5B")
    itok, inst = _qwen("Qwen/Qwen2.5-0.5B-Instruct")
    rows = []
    for q in CHAT_PROMPTS:
        print(q, flush=True)
        chat = itok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        r = {"prompt": q,
             "base": _gen(btok, base, q, 2, 0.7, seed=10),
             "urial": _gen(btok, base, URIAL.replace("{q}", q), 1, 0.0, stop=["```"])[0].strip(),
             "instruct": _gen(itok, inst, chat, 2, 0.7, seed=10)}
        # token-analyse op het (greedy) instruct-antwoord
        ans = _gen(itok, inst, chat, 1, 0.0, max_new=150)[0]
        c_ids = itok(chat).input_ids
        a_ids = itok(ans).input_ids
        ids = torch.tensor([c_ids + a_ids])
        with torch.no_grad():
            lb = torch.log_softmax(base(ids).logits[0], -1)
            li = torch.log_softmax(inst(ids).logits[0], -1)
        toks = []
        for j, t in enumerate(a_ids):
            i = len(c_ids) - 1 + j
            rank = int((lb[i] > lb[i, t]).sum().item()) + 1
            top_b = int(lb[i].argmax())
            toks.append({"t": itok.decode([t]), "rank": rank, "p_base": round(float(lb[i, t].exp()), 4),
                         "p_inst": round(float(li[i, t].exp()), 4), "base_top": itok.decode([top_b])})
        r["analysis"] = {"answer": ans, "tokens": toks}
        rows.append(r)
        dump("sftdiff", rows)



def exp_gpt2play():
    """Voegt GPT-2 (2019, puur tekstvoorspeller) toe aan de speeltuindata van exp_sftdiff."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok, model = AutoTokenizer.from_pretrained("gpt2"), AutoModelForCausalLM.from_pretrained("gpt2").eval()
    rows = json.loads((RES / "m2_sftdiff.json").read_text(encoding="utf-8"))
    for r in rows:
        r["gpt2"] = _gen(tok, model, r["prompt"], 2, 0.7, max_new=80, seed=10)
        print(r["prompt"], r["gpt2"], flush=True)
    dump("sftdiff", rows)


# --------------------------------------------------------------------------- reward model (HH-RLHF)
def _hh(split, n, seed):
    from datasets import load_dataset
    d = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    idx = random.Random(seed).sample(range(len(d)), n)
    return [d[i] for i in idx]


def _last_reply(dialog):
    return dialog.rsplit("\n\nAssistant:", 1)[-1].strip()


@torch.no_grad()
def _features(tok, model, texts, max_len=256, bs=16):
    """Laatste verborgen toestand van het laatste token (links afgekapt): één vector per tekst."""
    tok.padding_side, tok.truncation_side = "left", "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feats = []
    for i in range(0, len(texts), bs):
        enc = tok([t + tok.eos_token for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        h = model(**enc, output_hidden_states=True).hidden_states[-1][:, -1]
        feats.append(h.float())
        if i % 320 == 0:
            print(f"  features {i}/{len(texts)}", flush=True)
    return torch.cat(feats)


def exp_rm():
    """Reward model = lineaire kop op bevroren SmolLM2-135M-features, getraind met het
    Bradley-Terry-verlies op voorkeursparen uit Anthropic HH-RLHF (helpful-base)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "HuggingFaceTB/SmolLM2-135M"
    tok, lm = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval()
    tr, te = _hh("train", 2000, seed=0), _hh("test", 600, seed=0)
    F = {}
    for nm, ds in (("tr", tr), ("te", te)):
        F[nm + "_c"] = _features(tok, lm, [r["chosen"] for r in ds])
        F[nm + "_r"] = _features(tok, lm, [r["rejected"] for r in ds])
    torch.save(F, DATA / "m2_rm_features.pt")
    mu, sd = F["tr_c"].mean(0), F["tr_c"].std(0) + 1e-6
    norm = lambda x: (x - mu) / sd

    def fit(n):
        torch.manual_seed(0)
        w = torch.zeros(F["tr_c"].shape[1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=1e-2, weight_decay=0)
        Xc, Xr = norm(F["tr_c"][:n]), norm(F["tr_r"][:n])
        for _ in range(300):
            loss = -torch.nn.functional.logsigmoid(Xc @ w - Xr @ w).mean() + 1e-2 * (w ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return w.detach()

    curve = []
    for n in (62, 125, 250, 500, 1000, 2000):
        w = fit(n)
        acc_tr = ((norm(F["tr_c"][:n]) @ w) > (norm(F["tr_r"][:n]) @ w)).float().mean().item()
        acc_te = ((norm(F["te_c"]) @ w) > (norm(F["te_r"]) @ w)).float().mean().item()
        curve.append({"n": n, "train_acc": acc_tr, "test_acc": acc_te})
        print(curve[-1], flush=True)
    w = fit(2000)
    sc, sr = (norm(F["te_c"]) @ w).tolist(), (norm(F["te_r"]) @ w).tolist()
    lc = [len(_last_reply(r["chosen"])) for r in te]
    lr_ = [len(_last_reply(r["rejected"])) for r in te]
    longer = [a > b for a, b in zip(lc, lr_) if a != b]
    dump("rm", {"curve": curve, "score_chosen": sc, "score_rejected": sr, "len_chosen": lc, "len_rejected": lr_,
                "length_heuristic_acc": float(np.mean(longer)), "n_test": len(te)})


# --------------------------------------------------------------------------- DPO
def _to_messages(dialog):
    parts = re.split(r"\n\n(Human|Assistant): ", "\n\n" + dialog.strip())
    msgs = []
    for role, text in zip(parts[1::2], parts[2::2]):
        msgs.append({"role": "user" if role == "Human" else "assistant", "content": text.strip()})
    return msgs


def _dpo_pairs(tok, split, n, seed, max_len=192):
    out = []
    for r in _hh(split, 6 * n, seed):
        c, rj = _to_messages(r["chosen"]), _to_messages(r["rejected"])
        if len(c) < 2 or c[:-1] != rj[:-1] or c[-1]["role"] != "assistant" or c[-1]["content"] == rj[-1]["content"]:
            continue
        prompt = tok.apply_chat_template(c[:-1], tokenize=False, add_generation_prompt=True)
        p = tok(prompt).input_ids
        a = tok(c[-1]["content"] + "<|im_end|>").input_ids
        b = tok(rj[-1]["content"] + "<|im_end|>").input_ids
        if len(p) + max(len(a), len(b)) <= max_len:
            out.append((p, a, b))
        if len(out) == n:
            break
    return out


def _seq_logp(model, prompt_ids, ans_ids):
    """Som van log p(antwoord-tokens | prompt) per paar; batch via padding."""
    seqs = [p + a for p, a in zip(prompt_ids, ans_ids)]
    L = max(map(len, seqs))
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L)
    for i, (p, a) in enumerate(zip(prompt_ids, ans_ids)):
        ids[i, :len(p) + len(a)] = torch.tensor(p + a)
        mask[i, len(p):len(p) + len(a)] = 1
    att = (torch.arange(L)[None] < torch.tensor([len(s) for s in seqs])[:, None]).long()
    lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
    tgt = lp.gather(-1, ids[:, 1:, None])[..., 0]
    return (tgt * mask[:, 1:]).sum(-1)


def exp_dpo():
    """Een kleine DPO-run: SmolLM2-135M-Instruct met LoRA op HH-RLHF-paren, beta = 0.1."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             lora_dropout=0.0))
    train_pairs, test_pairs = _dpo_pairs(tok, "train", 600, 1), _dpo_pairs(tok, "test", 100, 1)
    beta, B, steps = 0.1, 4, 150
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    def logps(pairs, grad):
        P, A, R = zip(*pairs)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            pc, pr = _seq_logp(model, P, A), _seq_logp(model, P, R)
        with torch.no_grad(), model.disable_adapter():
            rc, rr = _seq_logp(model, P, A), _seq_logp(model, P, R)
        return pc, pr, rc, rr

    def evaluate():
        model.eval()
        m, acc, dc, dr = [], [], [], []
        for i in range(0, len(test_pairs), 10):
            pc, pr, rc, rr = logps(test_pairs[i:i + 10], False)
            marg = beta * ((pc - rc) - (pr - rr))
            m += marg.tolist(); acc += ((marg > 0).float() + 0.5 * (marg == 0).float()).tolist()   # gelijk = half goed
            dc += (pc - rc).tolist(); dr += (pr - rr).tolist()
        model.train()
        return {"margin": float(np.mean(m)), "acc": float(np.mean(acc)),
                "dlogp_chosen": float(np.mean(dc)), "dlogp_rejected": float(np.mean(dr))}

    log, rng = [], random.Random(0)
    for step in range(steps + 1):
        if step % 25 == 0:
            ev = evaluate(); ev["step"] = step; log.append(ev)
            print(ev, flush=True)
            dump("dpo", {"beta": beta, "batch_pairs": B, "lr": 3e-4, "log": log, "n_test": len(test_pairs)})
        if step == steps:
            break
        batch = rng.sample(train_pairs, B)
        pc, pr, rc, rr = logps(batch, True)
        loss = -torch.nn.functional.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"step": step, "train_loss": loss.item()})


# --------------------------------------------------------------------------- GSM8K: test-time compute
def _boxed(s):
    m = re.findall(r"\\boxed\{([^{}]*)\}", s)
    x = m[-1] if m else (re.findall(r"-?\d[\d,]*\.?\d*", s) or [None])[-1]
    if x is None:
        return None
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def exp_gsm8k(n_problems=40, n_samples=16):
    """Qwen2.5-0.5B-Instruct op GSM8K: greedy, plus 16 samples per vraag (T = 0,7)."""
    from datasets import load_dataset
    tok, model = _qwen("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(0).sample(range(len(ds)), n_problems)
    path = RES / "m2_gsm8k.json"
    rows = json.loads(path.read_text()) if path.exists() else []          # hervat waar je was
    for k, i in enumerate(idx):
        if k < len(rows):
            continue
        q, gold = ds[i]["question"], float(ds[i]["answer"].split("####")[-1].replace(",", "").strip())
        chat = tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
            greedy = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            torch.manual_seed(k)
            s = model.generate(ids.repeat(n_samples, 1), max_new_tokens=320, do_sample=True, temperature=0.7,
                               top_p=0.95, top_k=0, pad_token_id=tok.eos_token_id)
        samples = [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in s]
        rows.append({"q": q, "gold": gold, "greedy": _boxed(greedy), "answers": [_boxed(x) for x in samples],
                     "len": [len(tok(x).input_ids) for x in samples], "example": samples[0] if k < 3 else None})
        print(k, gold, rows[-1]["greedy"], rows[-1]["answers"], f"{time.time() - t0:.0f}s", flush=True)
        dump("gsm8k", rows)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[f"exp_{name}"]()
