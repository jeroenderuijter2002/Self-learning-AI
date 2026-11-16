"""
Genereert alle figuren voor module 1 en de minis, met echte data waar mogelijk
(GPT-2 via Hugging Face, de GPT-4-tokenizer via tiktoken, je eigen BPE uit O1).

    pip install numpy matplotlib torch transformers tiktoken
    python tools/make_figures.py                 # alles (~3 min op een laptop)
    python tools/make_figures.py gpt2_heads rope # alleen deze figuren

Beschikbare namen: zie ALL onderaan.
"""
import sys, time, re, os, urllib.request
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = str(Path(__file__).resolve().parents[1])
DATA = Path(__file__).resolve().parent / "data"
FIG1 = f"{ROOT}/modules/01-transformers/figures"
FIGM = f"{ROOT}/minis/figures"
os.makedirs(FIG1, exist_ok=True); os.makedirs(FIGM, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def data_file(name, url):
    DATA.mkdir(exist_ok=True)
    p = DATA / name
    if not p.exists():
        print("download", url); urllib.request.urlretrieve(url, p)
    return p


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", path)


def caption(fig, text):
    pass  # onderschrift staat in de notes


# ---------------------------------------------------------------- GPT-2 (lazy)
_m = {}
def gpt2():
    if not _m:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _m["tok"] = GPT2TokenizerFast.from_pretrained("gpt2")
        _m["model"] = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        _m["torch"] = torch
    return _m["tok"], _m["model"], _m["torch"]


# ============================================================ 1.1 figures
def fig_tokens():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    lines = [
        ("Engels", "The quick brown fox jumps over the lazy dog."),
        ("Nederlands", "De snelle bruine vos springt over de luie hond."),
        ("Lang NL-woord", "ongezelligheidsfactor"),
        ("Getallen", "12345 + 67890 = 80235"),
        ("Hoofdletters / spaties", "hello Hello  HELLO"),
        ("strawberry", "How many r's in strawberry?"),
    ]
    fills = ["#cde2fb", "#fde0d2", "#c9eedf", "#fbe7b0"]
    def width(text):
        return 22 + sum(1.35 * len(enc.decode([t])) + 2.1 for t in enc.encode(text))
    W = max(width(t) for _, t in lines) + 1
    fig, ax = plt.subplots(figsize=(11 * W / 100, 4.9))
    ax.set_xlim(0, W); ax.set_ylim(len(lines) * 10, -2); ax.axis("off")
    for r, (label, text) in enumerate(lines):
        ids = enc.encode(text)
        y = r * 10 + 2
        ax.text(0, y + 2.6, label, fontsize=9.5, color=INK2, va="center")
        ax.text(0, y + 5.6, f"{len(ids)} tokens · {len(text)} tekens",
                fontsize=8.5, color=MUTED, va="center")
        x = 22
        for k, t in enumerate(ids):
            s = enc.decode([t]).replace(" ", "␣")
            w = 1.35 * len(s) + 1.6
            ax.add_patch(FancyBboxPatch((x, y), w, 5.2, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc=fills[k % 4], ec="none"))
            ax.text(x + w / 2, y + 2.6, s, ha="center", va="center",
                    fontsize=9.5, family="DejaVu Sans Mono")
            ax.text(x + w / 2, y + 7.1, str(t), ha="center", va="center",
                    fontsize=6.5, color=MUTED, family="DejaVu Sans Mono")
            x += w + 0.5
    ax.set_title("Hoe GPT-4's tokenizer (cl100k_base) tekst opknipt", pad=8)
    caption(fig, "Elke gekleurde blok is één token; het kleine getal is het token-id. ␣ = spatie (hoort bij het token). "
                 "Bron: tiktoken, echte output.")
    save(fig, f"{FIG1}/1.1-tokenization-voorbeelden.png")


def fig_bpe_steps():
    steps = [("aaabdaaabac", None), ("ZabdZabac", "aa → Z"), ("YbdYbac", "Za → Y"), ("XdXac", "Yb → X")]
    colors = {"Z": "#cde2fb", "Y": "#86b6ef", "X": "#3987e5"}
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(42, -2); ax.axis("off")
    for r, (s, rule) in enumerate(steps):
        y = r * 10
        x = 4
        for ch in s:
            fc = colors.get(ch, "#eeede9")
            ax.add_patch(FancyBboxPatch((x, y), 5, 6.5, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none"))
            ax.text(x + 2.5, y + 3.3, ch, ha="center", va="center", fontsize=12,
                    family="DejaVu Sans Mono", color="white" if ch == "X" else INK)
            x += 5.6
        ax.text(70, y + 3.3, f"{len(s)} tokens", va="center", color=INK2)
        if rule:
            ax.text(84, y + 3.3, "merge " + rule, va="center", color=INK, fontweight="bold")
    ax.set_title("BPE: steeds het vaakst voorkomende paar samenvoegen", pad=4)
    save(fig, f"{FIG1}/1.1-bpe-stappen.png")


def _train_bpe(ids, n_merges):
    sys.path.insert(0, f"{ROOT}/modules/01-transformers/oplossingen")
    from o1_bpe_oplossing import get_pair_counts, merge
    merges = []
    for k in range(n_merges):
        c = get_pair_counts(ids)
        p = max(c, key=c.get)
        ids = merge(ids, p, 256 + k)
        merges.append(p)
    return merges


def fig_bpe_compressie():
    sys.path.insert(0, f"{ROOT}/modules/01-transformers/oplossingen")
    from o1_bpe_oplossing import merge
    en = open(data_file("shakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"), encoding="utf-8").read()
    nl = open(data_file("havelaar.txt", "https://www.gutenberg.org/cache/epub/11024/pg11024.txt"), encoding="utf-8").read()
    nl = nl[nl.find("Ik ben makelaar"):] if "Ik ben makelaar" in nl else nl[20000:]
    train = list(en[:150_000].encode())
    ev_en = list(en[400_000:420_000].encode())
    ev_nl = list(nl[:20_000].encode())
    t0 = time.time()
    merges = _train_bpe(train, 768)
    print(f"bpe trained in {time.time()-t0:.0f}s")
    checkpoints = [0, 32, 64, 128, 256, 384, 512, 640, 768]
    res = {}
    for name, ev in [("en", ev_en), ("nl", ev_nl)]:
        ids, n0, out = ev, len(ev), []
        for k, p in enumerate(merges):
            if k in checkpoints: out.append(n0 / len(ids))
            ids = merge(ids, p, 256 + k)
        out.append(n0 / len(ids))
        res[name] = out
    V = [256 + c for c in checkpoints]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(V, res["en"], color=BLUE, marker="o", ms=5)
    ax.plot(V, res["nl"], color=ORANGE, marker="o", ms=5)
    ax.text(V[-1] + 15, res["en"][-1], "Engels\n(zelfde soort tekst\nals training)", color=INK, va="center", fontsize=9.5)
    ax.text(V[-1] + 15, res["nl"][-1], "Nederlands\n(Max Havelaar)", color=INK, va="center", fontsize=9.5)
    ax.set_xlim(230, 1230)
    ax.set_xlabel("vocabulairegrootte (256 bytes + aantal merges)")
    ax.set_ylabel("bytes per token  (hoger = beter)")
    ax.set_title("Jouw BPE uit O1, getraind op Shakespeare")
    caption(fig, "Echte meting met de O1-oplossing: getraind op 150k tekens Shakespeare, geëvalueerd op ongeziene Engelse "
                 "en Nederlandse tekst. Een Engels getrainde tokenizer comprimeert Nederlands slechter → meer tokens → duurder.")
    save(fig, f"{FIG1}/1.1-bpe-compressie.png")


def fig_talen():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    texts = {
        "Engels": "Artificial intelligence is changing how companies make decisions. Large language models can read documents, write code and answer questions, but they still make mistakes that people need to check.",
        "Nederlands": "Kunstmatige intelligentie verandert hoe bedrijven beslissingen nemen. Grote taalmodellen kunnen documenten lezen, code schrijven en vragen beantwoorden, maar ze maken nog steeds fouten die mensen moeten controleren.",
        "Duits": "Künstliche Intelligenz verändert, wie Unternehmen Entscheidungen treffen. Große Sprachmodelle können Dokumente lesen, Code schreiben und Fragen beantworten, machen aber immer noch Fehler, die Menschen überprüfen müssen.",
        "Grieks": "Η τεχνητή νοημοσύνη αλλάζει τον τρόπο που οι εταιρείες παίρνουν αποφάσεις. Τα μεγάλα γλωσσικά μοντέλα μπορούν να διαβάζουν έγγραφα, να γράφουν κώδικα και να απαντούν σε ερωτήσεις, αλλά εξακολουθούν να κάνουν λάθη.",
    }
    names = list(texts)
    n = [len(enc.encode(texts[k])) for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, n, color=BLUE, height=0.62)
    for yi, ni in zip(y, n):
        ax.text(ni + 2, yi, f"{ni}  ({ni / n[0]:.1f}×)", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y, names); ax.grid(axis="y", visible=False)
    ax.set_xlabel("aantal tokens (GPT-4 tokenizer)"); ax.set_xlim(0, max(n) * 1.25)
    ax.set_title("Dezelfde alinea, vertaald: hoeveel tokens kost het?")
    caption(fig, "Echte telling met tiktoken (cl100k_base). Meer tokens = meer rekenwerk, hogere API-kosten en minder tekst in het context window.")
    save(fig, f"{FIG1}/1.1-tokens-per-taal.png")


# ============================================================ 1.2 figures
def fig_attention_stappen():
    toks = ["De", "kat", "zat", "op"]
    rng = np.random.default_rng(3)
    S = np.round(rng.normal(0, 1.4, (4, 4)), 1)
    mask = np.tril(np.ones((4, 4), bool))
    Sm = np.where(mask, S, -np.inf)
    W = np.exp(Sm - Sm.max(1, keepdims=True)); W = W / W.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    panels = [(S, "1. scores  QKᵀ/√d", True), (Sm, "2. + causal mask", True), (W, "3. softmax per rij", False)]
    for ax, (M, title, raw) in zip(axes, panels):
        vis = np.where(np.isfinite(M), M, np.nan)
        if raw:
            ax.imshow(np.zeros_like(M), cmap=SEQ, vmin=0, vmax=1)
        else:
            ax.imshow(vis, cmap=SEQ, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#eeede9", ec=SURF, lw=2))
                    ax.text(j, i, "−∞", ha="center", va="center", color=MUTED, fontsize=10)
                else:
                    dark = (not raw) and v > 0.55
                    ax.text(j, i, f"{v:.2f}" if not raw else f"{v:.1f}", ha="center", va="center",
                            color="white" if dark else INK, fontsize=10)
        ax.set_xticks(range(4), toks); ax.set_yticks(range(4), toks)
        ax.tick_params(length=0); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("key (kijkt naar →)", fontsize=9)
    axes[0].set_ylabel("query (token dat kijkt)", fontsize=9)
    fig.suptitle("Attention in drie stappen", x=0.01, ha="left", fontweight="bold", fontsize=12.5, y=1.02)
    caption(fig, "Rekenvoorbeeld met willekeurige scores. Na de softmax telt elke rij op tot 1, en 'De' kan alleen naar zichzelf kijken.")
    save(fig, f"{FIG1}/1.2-attention-stappen.png")


def fig_gpt2_heads():
    tok, model, torch = gpt2()
    text = "When Mary and John went to the store, John gave a drink to"
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions
    A = torch.stack([a[0] for a in att]).numpy()  # (L, H, T, T)
    T = A.shape[-1]
    toks = [tok.decode([t]).replace(" ", "␣") for t in ids[0]]
    prev = np.array([[np.mean([A[l, h, i, i - 1] for i in range(1, T)]) for h in range(12)] for l in range(12)])
    sink = np.array([[A[l, h, 3:, 0].mean() for h in range(12)] for l in range(12)])
    mary = toks.index("␣Mary")
    johns = [i for i, t in enumerate(toks) if t == "␣John"]
    ioi = A[:, :, -1, mary] - A[:, :, -1, johns].max(-1)
    picks = [("previous-token head", np.unravel_index(prev.argmax(), prev.shape)),
             ("'attention sink' (eerste token)", np.unravel_index(sink[1:].argmax(), sink[1:].shape)),
             ("naam-verplaatser (kijkt naar Mary)", np.unravel_index(ioi.argmax(), ioi.shape))]
    picks[1] = (picks[1][0], (picks[1][1][0] + 1, picks[1][1][1]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (title, (l, h)) in zip(axes, picks):
        ax.imshow(A[l, h], cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(T), toks, rotation=90, fontsize=8)
        ax.set_yticks(range(T), toks, fontsize=8)
        ax.tick_params(length=0); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{title}\nlaag {l}, head {h}", fontsize=10.5)
    fig.suptitle("Drie echte attention-heads uit GPT-2", x=0.01, ha="left", fontweight="bold", fontsize=12.5, y=1.0)
    caption(fig, "Rij = token dat kijkt, kolom = token waarnaar gekeken wordt; donkerder = meer gewicht. Heads automatisch geselecteerd "
                 "op hun patroon. Rechts: vanaf het laatste token ('to') kijkt dit head naar 'Mary', precies de naam die het model moet voorspellen.")
    save(fig, f"{FIG1}/1.2-gpt2-attention-heads.png")
    print("heads:", picks)


def fig_kwadratisch():
    T = np.logspace(2, 6, 200)
    d = 4096
    frac = T / (6 * d + T) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(T, frac, color=BLUE)
    ax.set_xscale("log")
    for t, lab in [(4096, "4k"), (32768, "32k"), (200_000, "200k"), (1_000_000, "1M")]:
        f = t / (6 * d + t) * 100
        ax.scatter([t], [f], color=BLUE, s=40, zorder=3, edgecolor=SURF, linewidth=1.5)
        ax.text(t * 0.82, f + 1.5, f"{lab}: {f:.0f}%", ha="right", fontsize=9, color=INK2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("contextlengte T (tokens, log-schaal)")
    ax.set_ylabel("% rekenwerk in QKᵀ en ×V")
    ax.set_title("Bij lange context domineert het kwadratische deel van attention")
    caption(fig, "Berekend per laag voor d_model = 4096: projecties + MLP kosten ~24·T·d² FLOPs, de attention-matrix ~4·T²·d. "
                 "Aandeel = T / (6d + T). Daarom is lange context duur en zoekt men naar efficiëntere attention.")
    save(fig, f"{FIG1}/1.2-kwadratische-kosten.png")


# ============================================================ 1.3 figures
def fig_params():
    tok, model, torch = gpt2()
    groups = {"token-embedding": 0, "positie-embedding": 0, "attention (12 lagen)": 0,
              "MLP (12 lagen)": 0, "LayerNorms": 0}
    for n, p in model.transformer.named_parameters():
        k = ("token-embedding" if "wte" in n else "positie-embedding" if "wpe" in n else
             "attention (12 lagen)" if ".attn." in n else "MLP (12 lagen)" if ".mlp." in n else "LayerNorms")
        groups[k] += p.numel()
    names = list(groups); vals = np.array([groups[k] for k in names]) / 1e6
    total = vals.sum()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=BLUE, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}M  ({v / total * 100:.0f}%)" if v > 0.1 else f"{v*1000:.0f}k", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y, names); ax.grid(axis="y", visible=False); ax.set_xlim(0, vals.max() * 1.3)
    ax.set_xlabel("parameters (miljoen)")
    ax.set_title(f"Waar zitten de {total:.0f}M parameters van GPT-2?")
    caption(fig, "Geteld uit het echte model. Het LM-head deelt zijn gewichten met de token-embedding (weight tying) en telt dus niet extra.")
    save(fig, f"{FIG1}/1.3-gpt2-parameters.png")


def fig_gelu():
    from math import erf
    x = np.linspace(-4, 4, 400)
    gelu = 0.5 * x * (1 + np.array([erf(v / np.sqrt(2)) for v in x]))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, np.maximum(0, x), color=MUTED, lw=1.6, ls="--")
    ax.plot(x, gelu, color=BLUE)
    ax.text(2.2, 3.1, "ReLU", color=INK2); ax.text(2.6, 1.9, "GELU", color=INK)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title("De niet-lineariteit in de MLP: GELU")
    caption(fig, "GELU is een 'zachte' ReLU: glad rond 0 en een klein negatief dipje. GPT-2 en BERT gebruiken GELU; Llama gebruikt de verwante SwiGLU.")
    save(fig, f"{FIG1}/1.3-gelu.png")


def fig_logit_lens():
    tok, model, torch = gpt2()
    prompt = "The capital of the Netherlands is"
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # 13: embeddings + 12 lagen (laatste al na ln_f)
    target = tok.encode(" Amsterdam")[0]
    probs, tops = [], []
    for l, h in enumerate(hs):
        x = h[0, -1]
        if l < len(hs) - 1:
            x = model.transformer.ln_f(x)
        p = torch.softmax(model.lm_head(x), -1)
        probs.append(p[target].item() * 100)
        tops.append(tok.decode([p.argmax().item()]))
    L = np.arange(len(hs))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(L, probs, color=BLUE, marker="o", ms=5)
    names = ["emb"] + [str(i) for i in range(1, 13)]
    ax.set_xticks(L, [f"{n}\n{t.strip()}" for n, t in zip(names, tops)], fontsize=8.5)
    ax.set_ylim(-0.5, max(probs) * 1.25)
    ax.set_xlabel("laag  (tweede regel: meest waarschijnlijke token op dat moment)")
    ax.set_ylabel("P(' Amsterdam')  in %")
    ax.set_title("'Logit lens': wat zou GPT-2 voorspellen na elke laag?")
    ax.text(0.01, 0.95, f"prompt: \"{prompt}\"", transform=ax.transAxes, fontsize=9.5, color=INK2, va="top")
    caption(fig, "Echte meting: de residual stream na elke laag direct door ln_f + unembedding gestuurd. Onder de as: het meest waarschijnlijke "
                 "token op dat moment. Het antwoord wordt laag voor laag opgebouwd in de residual stream.")
    save(fig, f"{FIG1}/1.3-logit-lens.png")
    print(list(zip(tops, np.round(probs, 2))))


# ============================================================ 1.4 figures
def fig_rope():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    base = np.array([1.0, 0.25])
    for ax, (theta, title) in zip(axes, [(0.6, "snelle frequentie (θ = 0,6)"), (0.12, "langzame frequentie (θ = 0,12)")]):
        ax.add_patch(Circle((0, 0), np.linalg.norm(base), fill=False, ec=GRID, lw=1.2))
        for m in range(8):
            a = m * theta
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            v = R @ base
            c = SEQ(0.35 + 0.6 * m / 7)
            ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
            ax.text(*(v * 1.13), f"m={m}", ha="center", va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=10.5)
    fig.suptitle("RoPE: dezelfde vector, gedraaid met hoek m·θ op positie m", x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    caption(fig, "Elk paar dimensies van q en k draait met een eigen frequentie. Het inproduct q_m·k_n hangt alleen af van de hoek "
                 "(m−n)·θ, dus van de relatieve afstand. Snelle paren 'zien' korte afstanden, langzame paren lange.")
    save(fig, f"{FIG1}/1.4-rope.png")


def fig_temperatuur_gpt2():
    tok, model, torch = gpt2()
    prompt = "The capital of the Netherlands is"
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top = torch.topk(logits, 8).indices
    labels = [tok.decode([i]).replace(" ", "␣") for i in top]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, T in zip(axes, [0.5, 1.0, 1.5]):
        p = torch.softmax(logits / T, -1)[top].numpy() * 100
        rest = 100 - p.sum()
        x = np.arange(len(labels))
        ax.bar(x, p, color=BLUE, width=0.7)
        for xi, pi in zip(x, p):
            ax.text(xi, pi + 0.6, f"{pi:.1f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8.5, family="DejaVu Sans Mono")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"T = {T}   (rest van vocab: {rest:.0f}%)", fontsize=10.5)
    axes[0].set_ylabel("kans (%)")
    fig.suptitle(f"GPT-2's top-8 voor \"{prompt} …\" bij drie temperaturen", x=0.01, ha="left", fontweight="bold", fontsize=12.5, y=1.03)
    caption(fig, "Echte logits van GPT-2 small. Lage T maakt de verdeling scherper (bijna greedy), hoge T vlakker: dan krijgt de lange staart "
                 "van ~50.000 overige tokens veel meer kans. Let op: ' Amsterdam' staat pas op plek 3; een 124M-model weet weinig.")
    save(fig, f"{FIG1}/1.4-temperatuur-gpt2.png")


def fig_kvcache_meting():
    tok, model, torch = gpt2()
    prompt = tok("The history of artificial intelligence", return_tensors="pt").input_ids
    Ns = [25, 50, 100, 200, 300]
    res = {False: [], True: []}
    model.generate(prompt, max_new_tokens=5, pad_token_id=tok.eos_token_id)  # warm-up
    for n in Ns:
        for uc in [False, True]:
            t0 = time.time()
            model.generate(prompt, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                           use_cache=uc, pad_token_id=tok.eos_token_id)
            res[uc].append(time.time() - t0)
        print(n, res[False][-1], res[True][-1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(Ns, res[False], color=ORANGE, marker="o", ms=5)
    ax.plot(Ns, res[True], color=BLUE, marker="o", ms=5)
    ax.text(Ns[-1] + 8, res[False][-1], f"zonder KV-cache\n{res[False][-1]:.1f} s", va="center", fontsize=9.5)
    ax.text(Ns[-1] + 8, res[True][-1], f"met KV-cache\n{res[True][-1]:.1f} s", va="center", fontsize=9.5)
    ax.set_xlim(0, 380)
    ax.set_xlabel("aantal gegenereerde tokens"); ax.set_ylabel("tijd (seconden)")
    ax.set_title("KV-cache gemeten: GPT-2 small op een CPU")
    caption(fig, "Echte meting (greedy decoding). Zonder cache groeit het werk per stap mee met de lengte → totaal ~kwadratisch. "
                 "Met cache is elke stap ongeveer even duur → totaal ~lineair. Jouw getallen in O4 zullen anders zijn, de vorm niet.")
    save(fig, f"{FIG1}/1.4-kvcache-meting.png")


def fig_kvcache_geheugen():
    T = np.linspace(0, 128_000, 300)
    L, dh, bytes_ = 80, 128, 2
    mha = 2 * L * T * 64 * dh * bytes_ / 1e9
    gqa = 2 * L * T * 8 * dh * bytes_ / 1e9
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T / 1000, mha, color=ORANGE)
    ax.plot(T / 1000, gqa, color=BLUE)
    ax.axhline(140, color=MUTED, lw=1, ls="--")
    ax.text(2, 146, "modelgewichten zelf (~70B × 2 bytes)", fontsize=9, color=INK2)
    ax.text(129, mha[-1], f"64 KV-heads (MHA)\n{mha[-1]:.0f} GB", va="center", fontsize=9.5)
    ax.text(129, gqa[-1], f"8 KV-heads (GQA)\n{gqa[-1]:.0f} GB", va="center", fontsize=9.5)
    ax.set_xlim(0, 160)
    ax.set_xlabel("contextlengte (×1000 tokens)"); ax.set_ylabel("KV-cache per gesprek (GB)")
    ax.set_title("Lange context kost geheugen: KV-cache van een 70B-model")
    caption(fig, "Berekend: 2 (K en V) × 80 lagen × T × KV-heads × 128 × 2 bytes (bf16), Llama-3-70B-achtige vorm. "
                 "Zonder GQA is de cache van één lang gesprek groter dan het hele model.")
    save(fig, f"{FIG1}/1.4-kvcache-geheugen.png")


# ============================================================ minis
def fig_m1_temperatuur():
    z = np.array([2.0, 1.0, 0.0])
    taus = [0.25, 0.5, 1, 2, 5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    w = 0.15
    for k, tau in enumerate(taus):
        p = np.exp(z / tau) / np.exp(z / tau).sum()
        ax.bar(np.arange(3) + (k - 2) * w, p, width=w * 0.9, color=SEQ(0.3 + 0.7 * k / 4), label=f"τ = {tau}")
    ax.set_xticks(range(3), ["token A\n(logit 2)", "token B\n(logit 1)", "token C\n(logit 0)"])
    ax.set_ylabel("kans"); ax.legend(ncol=1, fontsize=8.5, loc="upper right"); ax.grid(axis="x", visible=False)
    ax.set_title("Zelfde logits, andere temperatuur")
    ax = axes[1]
    tt = np.logspace(-1.5, 1.5, 200)
    H = []
    for tau in tt:
        p = np.exp(z / tau - (z / tau).max()); p /= p.sum(); H.append(-(p * np.log(p + 1e-12)).sum())
    ax.plot(tt, H, color=BLUE); ax.set_xscale("log")
    ax.axhline(np.log(3), color=MUTED, ls="--", lw=1); ax.text(0.04, np.log(3) + 0.03, "max = ln 3", fontsize=9, color=INK2)
    ax.set_xlabel("temperatuur τ (log)"); ax.set_ylabel("entropie (nats)")
    ax.set_title("Onzekerheid groeit met τ")
    save(fig, f"{FIGM}/M1-temperatuur.png")


def fig_m2_wortel():
    rng = np.random.default_rng(0)
    ds = [4, 16, 64, 256, 1024]
    raw, sc = [], []
    for d in ds:
        Q, K = rng.normal(size=(64, 16, d)), rng.normal(size=(64, 16, d))
        S = np.einsum("bid,bjd->bij", Q, K)
        for arr, s in [(raw, S), (sc, S / np.sqrt(d))]:
            e = np.exp(s - s.max(-1, keepdims=True)); w = e / e.sum(-1, keepdims=True)
            arr.append(w.max(-1).mean())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ds, raw, color=ORANGE, marker="o", ms=5)
    ax.plot(ds, sc, color=BLUE, marker="o", ms=5)
    ax.text(ds[-1] * 1.15, raw[-1], "zonder schaling", va="center", fontsize=9.5)
    ax.text(ds[-1] * 1.15, sc[-1], "met /√d_k", va="center", fontsize=9.5)
    ax.axhline(1 / 16, color=MUTED, ls="--", lw=1); ax.text(4, 1 / 16 + 0.03, "uniform (1/16)", fontsize=9, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xlim(3, 4000); ax.set_ylim(0, 1.05)
    ax.set_xlabel("d_k (log)"); ax.set_ylabel("gemiddeld grootste attention-gewicht")
    ax.set_title("Zonder schaling wordt attention one-hot naarmate d_k groeit")
    caption(fig, "Simulatie met 16 tokens en willekeurige q, k ~ N(0,1). Met schaling blijft de verdeling even 'zacht', ongeacht de dimensie.")
    save(fig, f"{FIGM}/M2-schaling.png")




def fig_embeddings_pca():
    tok, model, torch = gpt2()
    E = model.transformer.wte.weight.detach().numpy()
    groups = {
        "maanden": [" January", " February", " March", " April", " May", " June", " July", " August", " September", " October", " November", " December"],
        "getallen": [" one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten"],
        "landen": [" France", " Germany", " Italy", " Spain", " Japan", " China", " Canada", " Brazil", " Russia", " India"],
    }
    cols = {"maanden": BLUE, "getallen": ORANGE, "landen": AQUA}
    ids, lab = [], []
    for g, ws in groups.items():
        for w in ws:
            t = tok.encode(w)
            if len(t) == 1: ids.append(t[0]); lab.append((g, w.strip()))
    def pca(X):
        X = X - X.mean(0); _, _, Vt = np.linalg.svd(X, full_matrices=False); return X @ Vt[:2].T
    P = pca(E[ids])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for g, c in cols.items():
        pts = np.array([p for (gg, _), p in zip(lab, P) if gg == g])
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
        cx, cy = pts.mean(0)
        ax.annotate(g, (cx, cy), xytext=(14, 14), textcoords="offset points", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.set_title("Drie woordgroepen: drie clusters", fontsize=11)
    ax = axes[1]
    sel = [(i, w) for i, (g, w) in zip(ids, lab) if g == "maanden"]
    Q = pca(E[[i for i, _ in sel]])
    ax.plot(np.r_[Q[:, 0], Q[0, 0]], np.r_[Q[:, 1], Q[0, 1]], color=GRID, lw=1.2, zorder=1)
    ax.scatter(Q[:, 0], Q[:, 1], s=50, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    for (_, w), (x, y) in zip(sel, Q):
        ax.annotate(w, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PCA 1 (alleen maanden)"); ax.set_ylabel("PCA 2")
    ax.set_title("Ingezoomd op de maanden (lijn = kalendervolgorde)", fontsize=11)
    fig.suptitle("Echte GPT-2 token-embeddings, geprojecteerd naar 2D", x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    save(fig, f"{FIG1}/1.1-embeddings-pca.png")


ALL = {k[4:]: v for k, v in globals().items() if k.startswith("fig_")}
if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
