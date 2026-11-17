"""
Oplossing O1: Meet je eigen scaling law
=======================================
(Kijk pas als je vastzit.)

Deel A: vul de TODO's in en draai

    python o1_scaling.py            # draait de tests (een paar seconden)

Deel B: het echte experiment (ongeveer 15-30 min op een laptop-CPU)

    python o1_scaling.py --train    # traint 5 mini-GPT's en maakt een plot

Conventies:
    N = aantal parameters ZONDER embeddings (zoals Kaplan et al. 2020)
    D = aantal trainingstokens
    C = training-compute in FLOPs
"""
import math
import sys

import numpy as np


def count_params(n_layer: int, n_embd: int) -> int:
    """Aantal parameters zonder embeddings van de GPT in minigpt.py.

    Tel per block: de qkv-projectie (met bias), de output-projectie (met bias),
    de MLP (d -> 4d -> d, met biases) en twee LayerNorms (elk gewicht + bias).
    Tel daarna de laatste LayerNorm (ln_f) erbij. De head deelt zijn gewichten
    met de token-embedding (weight tying) en telt dus niet mee.

    Hint: je zou op 12*d^2 + 13*d per block moeten uitkomen. Waarom 12?
    """
    d = n_embd
    per_block = (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + 2 * (2 * d)
    return n_layer * per_block + 2 * d


def training_flops(n_params: float, n_tokens: float) -> float:
    """C ≈ 6·N·D: 2 FLOPs per parameter per token vooruit, 4 terug (notes 2.1)."""
    return 6.0 * n_params * n_tokens


def fit_power_law(N, L, n_grid: int = 400):
    """Fit L(N) = E + A * N^(-alpha) aan gemeten punten.

    Aanpak (zonder scipy): als je E kent, is log(L - E) = log(A) - alpha*log(N)
    een rechte lijn, en die fit je met kleinste kwadraten (np.polyfit).
    Probeer dus een rooster van waarden voor E tussen 0 en net onder min(L),
    fit voor elke E de rechte lijn, en kies de E met de kleinste fout op L zelf.

    Returns: (E, A, alpha)
    """
    N, L = np.asarray(N, float), np.asarray(L, float)
    best = None
    for E in np.linspace(0.0, L.min() * 0.999, n_grid):
        slope, intercept = np.polyfit(np.log(N), np.log(L - E), 1)
        pred = E + np.exp(intercept) * N ** slope
        err = np.sum((pred - L) ** 2)
        if best is None or err < best[0]:
            best = (err, E, np.exp(intercept), -slope)
    return best[1], best[2], best[3]


def chinchilla_loss(N, D, E, A, B, alpha, beta):
    """L(N, D) = E + A/N^alpha + B/D^beta  (Hoffmann et al. 2022, notes 2.2)."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def chinchilla_optimal(C, E, A, B, alpha, beta):
    """Optimale (N, D) bij budget C = 6·N·D.

    Leid af door D = C/(6N) in te vullen en d/dN op nul te zetten (notes 2.2):
        N_opt = G * (C/6)^(beta/(alpha+beta)),  G = (alpha*A / (beta*B))^(1/(alpha+beta))
        D_opt = C / (6 * N_opt)
    Returns: (N_opt, D_opt)
    """
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N_opt = G * (C / 6.0) ** (beta / (alpha + beta))
    return N_opt, C / (6.0 * N_opt)


# ---------------------------------------------------------------------------
# Deel B: het experiment
# ---------------------------------------------------------------------------
SIZES = [(1, 32), (2, 64), (3, 96), (4, 128), (4, 192)]


def run_experiment(max_steps=2000):
    """Traint de modellen uit SIZES met hetzelfde schema en dezelfde data, en plot."""
    import matplotlib.pyplot as plt
    import torch
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'opdrachten'))
    from minigpt import Config, load_shakespeare, train
    tr, va, _, _ = load_shakespeare()
    Ns, Ls, logs = [], [], []
    for L_, d in SIZES:
        cfg = Config(n_layer=L_, n_embd=d, n_head=4, block_size=64)
        print(f"\n=== n_layer={L_}, n_embd={d}, N={count_params(L_, d):,} ===")
        model, log = train(cfg, tr, va, max_steps=max_steps)
        assert model.n_params() == count_params(L_, d), "count_params klopt niet met het echte model!"
        Ns.append(model.n_params()); Ls.append(min(x["val"] for x in log)); logs.append(log)
    E, A, alpha = fit_power_law(Ns, Ls)
    print(f"\nFit: L(N) = {E:.3f} + {A:.1f} * N^-{alpha:.3f}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for n, log in zip(Ns, logs):
        C = [training_flops(n, x["tokens"]) for x in log[1:]]
        a1.plot(C, [x["val"] for x in log[1:]], label=f"{n:,}")
    a1.set_xscale("log"); a1.set_xlabel("C (FLOPs)"); a1.set_ylabel("val loss"); a1.legend(title="N")
    nn = np.logspace(np.log10(min(Ns) / 2), np.log10(max(Ns) * 4), 100)
    a2.plot(nn, E + A * nn ** -alpha, "--", color="gray")
    a2.scatter(Ns, Ls); a2.set_xscale("log"); a2.set_xlabel("N"); a2.set_ylabel("beste val loss")
    plt.tight_layout(); plt.savefig("o1_scaling.png", dpi=150)
    print("Plot opgeslagen als o1_scaling.png")


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
def _tests():
    # parameters tellen: vergelijk met de echte modellen uit tools/results/m2_scaling.json
    assert count_params(1, 32) == 12_768, count_params(1, 32)
    assert count_params(4, 128) == 793_344
    assert count_params(6, 256) == 4_739_072
    # GPT-2 small (12 lagen, d=768) heeft ~85M parameters zonder embeddings
    assert abs(count_params(12, 768) - 85_056_000) < 50_000

    # FLOPs: GPT-3 (175B, 300B tokens) ≈ 3,15e23
    assert abs(training_flops(175e9, 300e9) - 3.15e23) / 3.15e23 < 0.01

    # power-law-fit op synthetische data met bekende parameters
    rng = np.random.default_rng(0)
    N = np.logspace(4, 7, 8)
    L = 1.5 + 30 * N ** -0.3
    E, A, alpha = fit_power_law(N, L)
    assert abs(E - 1.5) < 0.03 and abs(alpha - 0.3) < 0.03, (E, A, alpha)
    L_noisy = L * (1 + 0.003 * rng.normal(size=L.shape))
    E2, _, alpha2 = fit_power_law(N, L_noisy)
    assert abs(alpha2 - 0.3) < 0.08, alpha2

    # Chinchilla: de gesloten oplossing moet het minimum op de isoFLOP-curve zijn
    P = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)   # Epoch AI (2024)
    C = 5.76e23                                                          # Gopher/Chinchilla-budget
    N_opt, D_opt = chinchilla_optimal(C, **P)
    assert abs(6 * N_opt * D_opt - C) / C < 1e-9, "N_opt en D_opt moeten samen het budget opmaken"
    grid = np.logspace(9, 12, 20001)
    numeric = grid[np.argmin(chinchilla_loss(grid, C / (6 * grid), **P))]
    assert abs(np.log(N_opt / numeric)) < 0.01, (N_opt, numeric)
    assert 15 < D_opt / N_opt < 25, "rond de 20 tokens per parameter"
    print(f"Chinchilla-optimum bij C = {C:.2e}: N ≈ {N_opt / 1e9:.0f}B, D ≈ {D_opt / 1e12:.2f}T "
          f"({D_opt / N_opt:.0f} tokens/parameter)")
    print("Alle tests geslaagd!")


if __name__ == "__main__":
    if "--train" in sys.argv:
        run_experiment()
    else:
        _tests()
