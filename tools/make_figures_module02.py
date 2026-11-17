"""
Maakt alle figuren voor module 2 (Van tekstvoorspeller naar assistent).

De meeste figuren lezen de resultaten van tools/m2_experiments.py uit tools/results/.
Die staan in de repo, dus je kunt de figuren meteen hermaken:

    pip install numpy matplotlib scipy transformers
    python tools/make_figures_module02.py                  # alles
    python tools/make_figures_module02.py scaling chinchilla

Wil je de metingen zelf opnieuw doen, draai dan eerst tools/m2_experiments.py.
Huisstijl: zie STIJLGIDS.md (zelfde kleuren als tools/make_figures.py).
"""
import json, math, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).resolve().parent / "results"
FIG = ROOT / "modules/02-training/figures"
FIG.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_COLORS = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq", SEQ_COLORS)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def load(name):
    return json.loads((RES / f"m2_{name}.json").read_text(encoding="utf-8"))


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", p.relative_to(ROOT))


def fmt_n(n):
    for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.0f}{s}" if v >= 10 else f"{v:.1f}{s}".replace(".0" + s, s)
    return str(int(n))


# ===================================================================== 2.1 pretraining
def fig_datamix():
    """LLaMA-1 pretraining-mix: Touvron et al. (2023), tabel 1."""
    rows = [("CommonCrawl (gefilterd)", 67.0, "webpagina's"), ("C4", 15.0, "webpagina's, anders gefilterd"),
            ("GitHub", 4.5, "code"), ("Wikipedia", 4.5, "20 talen"), ("Boeken", 4.5, "Gutenberg + Books3"),
            ("ArXiv", 2.5, "wetenschappelijke artikelen"), ("StackExchange", 2.0, "vraag en antwoord")]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    y = np.arange(len(rows))[::-1]
    cols = [BLUE, BLUE] + [ORANGE] + [AQUA] * 4
    ax.barh(y, [r[1] for r in rows], color=cols, height=0.62)
    for yi, (name, v, desc) in zip(y, rows):
        ax.text(v + 0.8, yi, f"{v:.1f}%".replace(".", ","), va="center", fontsize=9.5, color=INK)
        ax.text(v + 8.0, yi, desc, va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100); ax.set_xlabel("aandeel van de trainingstokens (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Waar leerde LLaMA (2023) van? 82% is web")
    save(fig, "2.1-datamix-llama.png")


def fig_trainingsrun():
    """Links: het lr-schema van de scaling-runs. Rechts: drie runs met verschillende lr."""
    lr = load("lr")
    sc = load("scaling")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1.35]})
    log = sc["runs"][-1]["log"]
    steps = np.arange(0, log[-1]["step"] + 1)
    import importlib.util
    spec = importlib.util.spec_from_file_location("minigpt", ROOT / "modules/02-training/opdrachten/minigpt.py")
    mg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg)
    a1.plot(steps, [mg.lr_at(s, steps[-1], 1.5e-3, 100) * 1e3 for s in steps], color=BLUE)
    a1.annotate("warmup\n(100 stappen)", xy=(100, 1.5), xytext=(330, 1.62), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    a1.text(1150, 0.95, "cosine decay\nnaar 10%", fontsize=9, color=INK2)
    a1.set_xlabel("trainingsstap"); a1.set_ylabel("learning rate (× 10⁻³)")
    a1.set_ylim(0, 1.8); a1.set_title("Het learning-rate-schema")
    colors = [BLUE, ORANGE, AQUA]
    for run, c in zip(lr, colors):
        st = [r["step"] for r in run["log"]]
        v = [r["val"] for r in run["log"]]
        a2.plot(st, v, color=c, lw=1.6)
        a2.text(st[-1] + 15, v[-1], run["name"], color=c, fontsize=9, va="center")
    a2.set_xlim(0, 1380); a2.set_ylim(1.4, 3.6)
    a2.set_xlabel("trainingsstap"); a2.set_ylabel("validatie-loss (nats per teken)")
    a2.set_title("Zelfde model, andere learning rate")
    save(fig, "2.1-trainingsrun.png")


# ===================================================================== 2.2 scaling laws
def power_fit(N, L):
    """Fit L = E + A * N^-alpha (kleinste kwadraten in log-ruimte voor het residu)."""
    from scipy.optimize import curve_fit
    f = lambda n, E, logA, a: E + np.exp(logA) * n ** (-a)
    p, _ = curve_fit(f, N, L, p0=(1.0, 2.0, 0.3), maxfev=20000)
    return p, f


def fig_scaling():
    sc = load("scaling")
    runs = sc["runs"]
    tok_per_step = sc["block_size"] * sc["batch_size"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.3))
    cols = [SEQ(0.4 + 0.6 * i / (len(runs) - 1)) for i in range(len(runs))]
    best = []
    for r, c in zip(runs, cols):
        C = np.array([6 * r["N_total"] * x["tokens"] for x in r["log"] if x["step"] > 0])
        v = np.array([x["val"] for x in r["log"] if x["step"] > 0])
        a1.plot(C, v, color=c, lw=1.6)
        a1.text(C[-1] * 1.3, v[-1] - 0.02, fmt_n(r["N"]), color=c, fontsize=8.5, va="center")
        best.append((C, v))
    a1.set_xscale("log"); a1.set_xlabel("training-compute C ≈ 6·N·D (FLOPs)")
    a1.set_ylabel("validatie-loss (nats per teken)"); a1.set_ylim(1.35, 3.0)
    a1.set_title("Elk model: loss tegen compute")
    N = np.array([r["N"] for r in runs], float)
    L = np.array([min(x["val"] for x in r["log"]) for r in runs])
    (E, logA, a), f = power_fit(N, L)
    nn = np.logspace(np.log10(N.min() / 2), np.log10(N.max() * 3), 100)
    a2.plot(nn, f(nn, E, logA, a), color=MUTED, lw=1.2, ls="--")
    a2.scatter(N, L, c=cols, s=46, zorder=3, edgecolor=INK, linewidth=0.5)
    for n, l in zip(N, L):
        a2.text(n * 1.15, l + 0.03, f"{l:.2f}".replace(".", ","), fontsize=8.5, color=INK2)
    a2.text(nn[52], f(nn[52], E, logA, a) + 0.16, f"fit: L(N) = {E:.2f} + {math.exp(logA):.1f}·N$^{{-{a:.2f}}}$".replace(".", ","),
            fontsize=10, color=INK)
    a2.set_xscale("log"); a2.set_xlabel("parameters N (zonder embeddings)")
    a2.set_ylabel("beste validatie-loss"); a2.set_ylim(1.35, 3.0)
    a2.set_title("Zes mini-GPT's: groter is voorspelbaar beter")
    save(fig, "2.2-scaling-eigen.png")
    return dict(E=E, A=math.exp(logA), alpha=a)


# Chinchilla-fit (Hoffmann et al. 2022, approach 3) en de replicatie door Epoch AI (Besiroglu et al. 2024)
HOFFMANN = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
EPOCH = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)


def chin_loss(N, D, p=EPOCH):
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]


def chin_opt(C, p=EPOCH):
    a, b = p["alpha"], p["beta"]
    G = (a * p["A"] / (b * p["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)


def fig_chinchilla():
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    budgets = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
    for i, C in enumerate(budgets):
        c = SEQ(0.3 + 0.7 * i / (len(budgets) - 1))
        N = np.logspace(7, 12.5, 400)
        D = C / (6 * N)
        L = chin_loss(N, D)
        ok = L < 4.0
        ax.plot(N[ok], L[ok], color=c, lw=1.6)
        No, Do = chin_opt(C)
        ax.scatter([No], [chin_loss(No, Do)], color=ORANGE, s=28, zorder=3)
        ax.text(No * 1.18, chin_loss(No, Do) + 0.015, f"10$^{{{int(round(math.log10(C)))}}}$", color=INK2, fontsize=9, ha="left", va="bottom")
    Cs = np.logspace(19, 24, 50)
    Ns = [chin_opt(C)[0] for C in Cs]
    ax.plot(Ns, [chin_loss(n, C / 6 / n) for n, C in zip(Ns, Cs)], color=ORANGE, lw=1, ls="--")
    No, Do = chin_opt(5.8e23)
    ax.text(1.1e7, 2.0, "één curve per compute-budget C (FLOPs)\noranje stip = beste modelgrootte", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_ylim(1.9, 3.7); ax.set_xlim(3e6, 3e12)
    ax.set_xlabel("modelgrootte N (parameters)"); ax.set_ylabel("voorspelde loss L(N, D)")
    ax.set_title("IsoFLOP-curves: bij vast budget is er een beste modelgrootte")
    save(fig, "2.2-chinchilla-isoflop.png")


MODELS = [  # (naam, parameters, trainingstokens, bron-sleutel)
    ("GPT-3", 175e9, 300e9), ("Gopher", 280e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
    ("LLaMA-7B", 6.7e9, 1.0e12), ("LLaMA-65B", 65.2e9, 1.4e12), ("Llama 2 70B", 70e9, 2.0e12),
    ("Llama 3 8B", 8e9, 15e12), ("Llama 3.1 405B", 405e9, 15.6e12),
    ("Pythia-1.4B", 1.4e9, 300e9), ("SmolLM2-135M", 135e6, 2e12), ("SmolLM2-1.7B", 1.7e9, 11e12),
]


def fig_tokens_per_param():
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    N = np.logspace(8, 12, 10)
    for ratio, ls in ((20, "--"), (200, ":"), (2000, ":")):
        ax.plot(N, ratio * N, color=MUTED if ratio != 20 else ORANGE, lw=1, ls=ls)
        if ratio == 20:
            ax.text(2.5e11, 20 * 2.5e11 * 1.3, "20 tokens/parameter\n(Chinchilla-optimaal)", color=ORANGE, fontsize=8.5, ha="right")
        else:
            ax.text(3.6e13 / ratio * 1.12, 3.6e13, f"{ratio:,} tokens/parameter".replace(",", "."), color=MUTED, fontsize=8.5, va="center")
    offs = {"GPT-3": (0.33, 0.68), "Gopher": (1.15, 1.0), "Chinchilla": (1.15, 0.75), "LLaMA-7B": (1.15, 0.8),
            "LLaMA-65B": (1.4, 1.17), "Llama 2 70B": (1.15, 1.25), "Llama 3 8B": (1.15, 1.0),
            "Llama 3.1 405B": (0.8, 1.0), "Pythia-1.4B": (0.8, 1.0), "SmolLM2-135M": (1.15, 1.0),
            "SmolLM2-1.7B": (1.15, 1.0)}
    for name, n, d in MODELS:
        c = BLUE if d / n > 60 else (AQUA if d / n > 10 else INK2)
        ax.scatter([n], [d], color=c, s=30, zorder=3)
        ox, oy = offs.get(name, (1.15, 1.0))
        lab = name.replace("3.1 405B", "3.1\n405B") if name.startswith("Llama 3.1") else name
        ax.text(n * ox, d * oy, f"{lab}  ({d / n:,.0f}×)".replace(",", "."), fontsize=8.5, color=INK, va="center",
                ha="right" if name in ("Pythia-1.4B", "Llama 3.1 405B") else "left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8e7, 2e12); ax.set_ylim(1e11, 6e13)
    ax.set_xlabel("parameters N"); ax.set_ylabel("trainingstokens D")
    ax.set_title("Hoeveel tokens per parameter? steeds meer dan 'Chinchilla-optimaal'")
    save(fig, "2.2-tokens-per-parameter.png")


def fig_emergentie():
    """Links: Pythia 70M-1.4B op optelsommen. Rechts: Qwen2.5 0.5B en 1.5B, 1 tot 8 cijfers."""
    py, dg = load("pythia"), load("digits")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ms = py["models"]
    N = [m["N"] for m in ms]
    for digits, ls in (("2", "-"), ("3", "--")):
        a1.plot(N, [m["tasks"][digits]["exact_match"] * 100 for m in ms], color=ORANGE, ls=ls, marker="o", ms=4)
        a1.plot(N, [m["tasks"][digits]["digit_acc"] * 100 for m in ms], color=BLUE, ls=ls, marker="o", ms=4)
    a1.text(N[-1], ms[-1]["tasks"]["2"]["digit_acc"] * 100 + 5, "cijfers goed", color=BLUE, ha="right", fontsize=9)
    a1.text(N[-1], 6, "exact match", color=ORANGE, ha="right", fontsize=9)
    a1.text(N[0], 92, "doorgetrokken: 2 cijfers\ngestippeld: 3 cijfers", color=INK2, fontsize=8.5, va="top")
    a1.set_xscale("log"); a1.set_xticks(N, [m["size"].upper() for m in ms], fontsize=8.5); a1.minorticks_off()
    a1.set_ylim(-3, 100); a1.set_xlabel("Pythia-model (zelfde data, zelfde volgorde)"); a1.set_ylabel("score (%)")
    a1.set_title("Pythia: optellen komt (nog) niet")
    cols = [BLUE, AQUA]
    for m, c in zip(dg["models"], cols):
        ks = sorted(int(k) for k in m["k"])
        em = [m["k"][str(k)]["exact_match"] * 100 for k in ks]
        da = [m["k"][str(k)]["digit_acc"] * 100 for k in ks]
        nm = m["name"].split("/")[-1].replace("Qwen2.5-", "")
        a2.plot(ks, em, color=c, marker="o", ms=4)
        a2.plot(ks, da, color=c, ls="--", lw=1.4)
        a2.text(ks[-1] + 0.15, em[-1], f"{nm}: hele getal goed", color=c, fontsize=8.5, va="center")
        a2.text(ks[-1] + 0.15, da[-1], f"{nm}: cijfers goed", color=c, fontsize=8.5, va="center")
    a2.set_xlim(0.7, 11.6); a2.set_ylim(-3, 103); a2.set_xticks(range(1, 9))
    a2.set_xlabel("aantal cijfers per getal"); a2.set_ylabel("score (%)")
    a2.set_title("Qwen2.5: dezelfde antwoorden, twee meetlatten")
    save(fig, "2.2-emergentie-pythia.png")


# ===================================================================== 2.3 SFT
def fig_chat_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    msgs = [{"role": "user", "content": "Wat is de hoofdstad van Nederland?"},
            {"role": "assistant", "content": "De hoofdstad van Nederland is Amsterdam."}]
    text = tok.apply_chat_template(msgs, tokenize=False, system="")
    # haal de standaard-systeemprompt weg zodat het plaatje leesbaar blijft
    text = text.split("<|im_end|>\n", 1)[1] if text.startswith("<|im_start|>system") else text
    ids = tok(text).input_ids
    pieces = [tok.decode([i]) for i in ids]
    start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    # bepaal per token of het bij het antwoord hoort (loss) of niet
    pos, mask = 0, []
    for p in pieces:
        mask.append(pos >= start)
        pos += len(p)
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(23, -1); ax.axis("off")
    x, y = 0.5, 1
    for p, m in zip(pieces, mask):
        special = p.startswith("<|")
        s = p.replace("\n", "↵").replace(" ", "␣")
        w = 0.95 * len(s) + 1.0 if not special else 0.78 * len(s) + 1.0
        if x + w > 99.5:
            x, y = 0.5, y + 7
        fc = "#cde2fb" if m else ("#eeede9" if not special else "#fbe7b0")
        ax.add_patch(FancyBboxPatch((x, y), w, 4.8, boxstyle="round,pad=0,rounding_size=0.6", fc=fc,
                                    ec=BLUE if m else "none", lw=0.8))
        ax.text(x + w / 2, y + 2.4, s, ha="center", va="center", fontsize=8 if special else 9,
                family="DejaVu Sans Mono", color=INK)
        x += w + 0.35
    ax.add_patch(FancyBboxPatch((0.5, 19.2), 2.2, 2.6, boxstyle="round,pad=0,rounding_size=0.4", fc="#fbe7b0", ec="none"))
    ax.text(3.2, 20.5, "speciaal token", fontsize=8.5, va="center", color=INK2)
    ax.add_patch(FancyBboxPatch((17, 19.2), 2.2, 2.6, boxstyle="round,pad=0,rounding_size=0.4", fc="#eeede9", ec="none"))
    ax.text(19.7, 20.5, "context: geen loss", fontsize=8.5, va="center", color=INK2)
    ax.add_patch(FancyBboxPatch((37, 19.2), 2.2, 2.6, boxstyle="round,pad=0,rounding_size=0.4", fc="#cde2fb", ec=BLUE, lw=0.8))
    ax.text(39.7, 20.5, "antwoord: hier wordt de loss berekend", fontsize=8.5, va="center", color=INK2)
    ax.set_title("Een gesprek is gewoon een document: de chat-template van Qwen2.5", pad=4)
    save(fig, "2.3-chat-template.png")


def fig_sft_tokenrang():
    rows = load("sftdiff")
    C1, C2, C3 = "#e9e8e3", "#f6d47a", "#f0915f"
    ex = next(r for r in rows if r["prompt"].startswith("Give me three tips"))
    toks = ex["analysis"]["tokens"]
    fig = plt.figure(figsize=(11.5, 6.4))
    ax = fig.add_axes([0.0, 0.3, 1.0, 0.65]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(52, -1)

    def col(rank):
        return C1 if rank == 1 else (C2 if rank <= 5 else C3)
    x, y = 0.5, 1
    for t in toks:
        s = t["t"]
        if s == "\n" or s == "\n\n":
            x, y = 0.5, y + 5.6 * (1 if s == "\n" else 1)
            continue
        s = s.replace("\n", "↵")
        w = 0.66 * len(s) + 1.0
        if x + w > 99.5:
            x, y = 0.5, y + 5.6
        ax.add_patch(FancyBboxPatch((x, y), w, 4.4, boxstyle="round,pad=0,rounding_size=0.5", fc=col(t["rank"]), ec="none"))
        ax.text(x + w / 2, y + 2.2, s, ha="center", va="center", fontsize=9, color=INK)
        x += w + 0.25
    ax.set_title(f"Qwen2.5-0.5B-Instruct: \"{ex['prompt']}\"  —  welke tokens zou het basismodel ook kiezen?", pad=2)
    # samenvatting over alle prompts
    a2 = fig.add_axes([0.06, 0.05, 0.6, 0.2])
    allr = [t["rank"] for r in rows for t in r["analysis"]["tokens"]]
    frac = [np.mean([r == 1 for r in allr]), np.mean([2 <= r <= 5 for r in allr]), np.mean([r > 5 for r in allr])]
    labels = ["rang 1: basismodel kiest hetzelfde", "rang 2–5", "rang > 5: echt anders"]
    left = 0
    for f, c, lab in zip(frac, [C1, C2, C3], labels):
        a2.barh([0], [f], left=left, color=c, height=0.6)
        a2.text(left + f / 2 if f > 0.06 else left + f + 0.01, 0, f"{f * 100:.0f}%", ha="center" if f > 0.06 else "left",
                va="center", fontsize=9.5, color=INK)
        left += f
    a2.set_xlim(0, 1); a2.set_yticks([]); a2.set_xticks([]); a2.grid(False)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title(f"Alle {len(rows)} prompts samen ({len(allr)} tokens)", fontsize=10.5)
    for i, (c, lab) in enumerate(zip([C1, C2, C3], labels)):
        a2.text(1.03, 0.25 - i * 0.3, "■", color=c, fontsize=12, va="center", transform=a2.transAxes)
        a2.text(1.07, 0.25 - i * 0.3, lab, fontsize=9, va="center", color=INK2, transform=a2.transAxes)
    save(fig, "2.3-sft-tokenrang.png")


# ===================================================================== 2.4 RLHF
def fig_rm():
    d = load("rm")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    n = [c["n"] for c in d["curve"]]
    a1.plot(n, [c["train_acc"] * 100 for c in d["curve"]], color=MUTED, marker="o", ms=4, lw=1.4)
    a1.plot(n, [c["test_acc"] * 100 for c in d["curve"]], color=BLUE, marker="o", ms=5)
    a1.text(n[-1] * 1.1, d["curve"][-1]["train_acc"] * 100, "train", color=MUTED, va="center", fontsize=9)
    a1.text(n[-1] * 1.1, d["curve"][-1]["test_acc"] * 100, "test", color=BLUE, va="center", fontsize=9)
    a1.axhline(50, color=GRID, lw=1.2); a1.text(n[-1], 51, "gokken", color=MUTED, fontsize=8.5, ha="right")
    a1.axhline(d["length_heuristic_acc"] * 100, color=ORANGE, lw=1, ls="--")
    a1.text(n[0], d["length_heuristic_acc"] * 100 + 1.4, "'kies gewoon het langste antwoord'", color=ORANGE, fontsize=8.5)
    a1.set_xscale("log"); a1.set_xticks(n, [str(x) for x in n]); a1.minorticks_off()
    a1.set_ylim(40, 100); a1.set_xlabel("aantal voorkeursparen in de training")
    a1.set_ylabel("paren goed geordend (%)"); a1.set_title("Reward model: hoe vaak kiest het als de mens?")
    sc, sr = np.array(d["score_chosen"]), np.array(d["score_rejected"])
    lc, lr_ = np.array(d["len_chosen"]), np.array(d["len_rejected"])
    s_all, l_all = np.concatenate([sc, sr]), np.concatenate([lc, lr_])
    a2.scatter(l_all, s_all, s=6, color=BLUE, alpha=0.25, edgecolor="none")
    bins = np.logspace(np.log10(max(5, l_all.min())), np.log10(l_all.max()), 12)
    idx = np.digitize(l_all, bins)
    mx = [np.median(l_all[idx == k]) for k in range(1, len(bins)) if (idx == k).sum() > 20]
    my = [np.median(s_all[idx == k]) for k in range(1, len(bins)) if (idx == k).sum() > 20]
    a2.plot(mx, my, color=ORANGE, lw=2)
    r = np.corrcoef(np.log(l_all + 1), s_all)[0, 1]
    a2.text(0.03, 0.93, f"correlatie(log lengte, score) = {r:.2f}".replace(".", ","), transform=a2.transAxes, fontsize=9.5, color=INK)
    a2.text(mx[-1], my[-1] + 0.4, "mediaan", color=ORANGE, fontsize=9, ha="right")
    a2.set_xscale("log"); a2.set_xlabel("lengte van het antwoord (tekens)"); a2.set_ylabel("reward-score")
    a2.set_title("Lange antwoorden krijgen meer beloning")
    save(fig, "2.4-reward-model.png")
    return r


# ===================================================================== 2.5 DPO
def fig_dpo():
    d = load("dpo")
    ev = [dict(x) for x in d["log"] if "acc" in x]
    for x in ev:                       # bij stap 0 is pi = pi_ref: elke marge is precies 0, dus 'half goed'
        if x["step"] == 0 and x["margin"] == 0:
            x["acc"] = 0.5
    tr = [x for x in d["log"] if "train_loss" in x]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    st = [x["step"] for x in ev]
    a1.plot(st, [x["dlogp_chosen"] for x in ev], color=BLUE, marker="o", ms=4)
    a1.plot(st, [x["dlogp_rejected"] for x in ev], color=ORANGE, marker="o", ms=4)
    a1.text(st[-1] + 3, ev[-1]["dlogp_chosen"], "gekozen antwoord", color=BLUE, va="center", fontsize=9)
    a1.text(st[-1] + 3, ev[-1]["dlogp_rejected"], "afgewezen antwoord", color=ORANGE, va="center", fontsize=9)
    a1.axhline(0, color=MUTED, lw=1)
    a1.set_xlim(0, st[-1] * 1.45)
    a1.set_xlabel("DPO-stap"); a1.set_ylabel("log π(y|x) − log π_ref(y|x)")
    a1.set_title("Hoe verschuift de kans? (test-paren)")
    a2.plot(st, [x["acc"] * 100 for x in ev], color=AQUA, marker="o", ms=4)
    a2.axhline(50, color=GRID, lw=1.2)
    a2.set_ylim(30, 100); a2.set_xlabel("DPO-stap"); a2.set_ylabel("impliciete reward kiest goed (%)")
    a2.set_title("Het verborgen reward model leert mee")
    save(fig, "2.5-dpo-run.png")


# ===================================================================== 2.6 reasoning
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_at_k(answers, gold, k, rng, reps=200):
    hits = 0
    for _ in range(reps):
        sub = [a for a in rng.choice(np.array(answers, dtype=object), k, replace=False) if a is not None]
        if not sub:
            continue
        vals, counts = np.unique(np.array(sub, float), return_counts=True)
        top = vals[counts == counts.max()]
        hits += rng.choice(top) == gold
    return hits / reps


def fig_gsm8k():
    rows = load("gsm8k")
    rng = np.random.default_rng(0)
    n = len(rows[0]["answers"])
    ks = [1, 2, 4, 8, 16]
    corr = lambda a, g: a is not None and abs(a - g) < 1e-6
    pk = [np.mean([pass_at_k(n, sum(corr(a, r["gold"]) for a in r["answers"]), k) for r in rows]) * 100 for k in ks]
    mj = [np.mean([majority_at_k(r["answers"], r["gold"], k, rng) for r in rows]) * 100 for k in ks]
    greedy = np.mean([corr(r["greedy"], r["gold"]) for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(ks, pk, color=BLUE, marker="o", ms=5)
    ax.plot(ks, mj, color=ORANGE, marker="o", ms=5)
    ax.axhline(greedy, color=MUTED, lw=1, ls="--")
    ax.text(ks[-1] * 1.08, pk[-1], f"pass@k: minstens één\ngoed antwoord ({pk[-1]:.0f}%)", color=BLUE, va="center", fontsize=9)
    ax.text(ks[-1] * 1.08, mj[-1], f"majority vote\n({mj[-1]:.0f}%)", color=ORANGE, va="center", fontsize=9)
    ax.text(ks[-1], greedy - 5, f"greedy, één poging ({greedy:.0f}%)", color=MUTED, fontsize=9, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.minorticks_off()
    ax.set_xlim(0.85, 40); ax.set_ylim(0, 100)
    ax.set_xlabel("aantal samples per vraag (k)"); ax.set_ylabel("vragen goed (%)")
    ax.set_title(f"Meer rekenen bij het antwoorden: Qwen2.5-0.5B-Instruct op {len(rows)} GSM8K-vragen")
    save(fig, "2.6-test-time-compute.png")
    return dict(pass_at_k=dict(zip(ks, pk)), majority=dict(zip(ks, mj)), greedy=greedy)


# ===================================================================== mini M5
def fig_m5_kl():
    """Het gesloten RLHF-optimum pi* ∝ pi_ref · exp(r/β) op een echte GPT-2-verdeling."""
    js = (ROOT / "interactief/data/sampling.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("["):js.rindex("]") + 1])
    d = next(x for x in data if x["prompt"].startswith("The capital of the Netherlands"))
    toks = [t for t, _ in d["top"]]
    z = np.array([v for _, v in d["top"]], float)
    # rest van de vocab als histogram: (centrum-logit, aantal)
    rest_c = np.array([c for c, _ in d["rest"]], float); rest_n = np.array([n for _, n in d["rest"]], float)
    m = z.max()
    w_top, w_rest = np.exp(z - m), rest_n * np.exp(rest_c - m)
    Zs = w_top.sum() + w_rest.sum()
    p_top, p_rest = w_top / Zs, w_rest / Zs
    i_ams = toks.index(" Amsterdam")
    r = np.zeros_like(p_top); r[i_ams] = 1.0          # beloning 1 voor " Amsterdam", anders 0
    betas = np.logspace(-1.3, 1.3, 120)
    P, KL = [], []
    for b in betas:
        q_top = p_top * np.exp(r / b); q_rest = p_rest.copy()
        Zq = q_top.sum() + q_rest.sum(); q_top /= Zq; q_rest /= Zq
        P.append(q_top[i_ams])
        kl = np.sum(q_top * np.log(q_top / p_top)) + np.sum(q_rest[q_rest > 0] * np.log(q_rest[q_rest > 0] / p_rest[q_rest > 0]))
        KL.append(kl)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    a1.plot(betas, np.array(P) * 100, color=BLUE)
    a1.axhline(p_top[i_ams] * 100, color=MUTED, lw=1, ls="--")
    a1.text(betas[-1], p_top[i_ams] * 100 + 3, f"GPT-2 zelf: {p_top[i_ams] * 100:.0f}%", color=MUTED, fontsize=9, ha="right")
    a1.set_xscale("log"); a1.set_ylim(0, 105)
    a1.set_xlabel("β (sterkte van de KL-rem)"); a1.set_ylabel("kans op ' Amsterdam' (%)")
    a1.set_title("Hoeveel duwt de beloning?")
    a2.plot(np.array(KL), np.array(P) * 100, color=ORANGE)
    for bb in (0.1, 0.3, 1, 3):
        k = np.argmin(abs(betas - bb))
        a2.scatter([KL[k]], [P[k] * 100], color=INK, s=16, zorder=3)
        a2.text(KL[k] + 0.08, P[k] * 100 - 4, f"β = {bb:g}", fontsize=8.5, color=INK2)
    a2.set_ylim(0, 105); a2.set_xlabel("KL(π* ‖ π_ref)  (nats)"); a2.set_ylabel("kans op ' Amsterdam' (%)")
    a2.set_title("Meer beloning kost meer afstand")
    save_m(fig, "M5-kl-rem.png")


def save_m(fig, name):
    p = ROOT / "minis/figures" / name
    fig.savefig(p, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", p.relative_to(ROOT))


ALL = {"datamix": fig_datamix, "trainingsrun": fig_trainingsrun, "scaling": fig_scaling,
       "chinchilla": fig_chinchilla, "tokens": fig_tokens_per_param, "emergentie": fig_emergentie,
       "template": fig_chat_template, "tokenrang": fig_sft_tokenrang, "rm": fig_rm, "dpo": fig_dpo,
       "gsm8k": fig_gsm8k, "m5": fig_m5_kl}

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        try:
            out = ALL[name]()
            if out is not None:
                print("  ", name, out)
        except FileNotFoundError as e:
            print(f"  {name}: resultaten ontbreken ({e.filename}); draai eerst tools/m2_experiments.py")
