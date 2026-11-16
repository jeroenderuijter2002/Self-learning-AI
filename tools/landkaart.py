import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys
from pathlib import Path
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "assets" / "landkaart.png")
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
ST = {"had": ("#c9eedf", "#1baf7a", INK, "solid"), "bezig": ("#fbe7b0", "#c98500", INK, "solid"),
      "repo": ("#cde2fb", "#2a78d6", INK, "solid"), "niet": ("#f4f3f0", "#b5b4ae", "#6b6a66", (0, (3, 2)))}
H_, R_, B_, N_ = "had", "repo", "bezig", "niet"
D = [  # titel, x, y, w, h, ncol, chips(tekst, status, badge)
 ("Taalmodellen & foundation models", 0, 62, 76, 11.5, 5, [
   ("Training: pretraining,\nRLHF, reasoning", R_, "M2"), ("AI engineering:\nRAG, agents, evals", R_, "M3 · stage"),
   ("Inference &\nefficiëntie", R_, "M4"), ("Interpretability\n& alignment", R_, "M7"), ("Klassieke NLP\n(n-grams, HMM's)", N_, "")]),
 ("Deep learning", 0, 42.5, 76, 18, 4, [
   ("Neurale netwerken\n& backprop", H_, ""), ("CNN's &\ncomputer vision", H_, "NNCV"), ("Transformers", R_, "M1"),
   ("Graph neural\nnetworks", R_, "M9"), ("Generatief: diffusion,\nmultimodaal", R_, "M6"), ("Spraak & audio", N_, ""),
   ("3D & graphics\n(NeRF)", N_, ""), ("Self-supervised &\nrepresentatie-leren", N_, "deels M2")]),
 ("Klassieke AI", 0, 12, 24, 29, 2, [
   ("Zoeken &\nplanning", R_, "M8"), ("Logica & kennis-\nrepresentatie", R_, "M8"), ("OR & beslissen\nonder onzekerheid", H_, "BSc"),
   ("Evolutionaire\nalgoritmen", H_, ""), ("Speltheorie &\nmulti-agent", N_, ""), ("Expert systems\n& fuzzy logic", N_, "")]),
 ("Machine learning", 25.5, 12, 27, 29, 2, [
   ("Supervised: regressie,\nSVM, bomen", H_, ""), ("Unsupervised:\nclustering, PCA", H_, ""), ("Bayesiaanse ML &\ngraphical models", B_, "5SSD0"),
   ("Gaussian processes\n& surrogaten", H_, "stage"), ("Gradient boosting\nin de praktijk", R_, "M14"), ("Causal inference\n& time series", R_, "M9"),
   ("Learning theory", N_, ""), ("Anomaly detection,\nAutoML, recsys", N_, "deels M13")]),
 ("Beslissen & besturen", 54, 12, 22, 29, 1, [
   ("MDP's & dynamic programming", H_, "SDT"), ("Regeltechniek & MPC", H_, "5LMB0"), ("Deep RL", B_, "M5 · deels gehad · thesis"), ("Robotica & embodied AI", N_, "")]),
 ("Wiskundige fundamenten", 0, 2, 76, 8.5, 4, [
   ("Lineaire algebra\n& calculus", H_, ""), ("Kansrekening\n& statistiek", H_, ""), ("Optimalisatie", H_, "4DM20"), ("Informatietheorie", B_, "5SSD0")]),
 ("Engineering", 78, 23, 22, 50, 1, [
   ("Software engineering", R_, "M10"), ("MLOps & LLMOps", R_, "M11"), ("Data engineering", R_, "M12 · SQL bezig"),
   ("Systeemontwerp\n& A/B-testen", R_, "M13"), ("GPU-programmering &\ndistributed training", N_, "")]),
 ("Randgebieden", 78, 2, 22, 19.5, 1, [
   ("Verantwoorde AI\n& regelgeving", H_, ""), ("Privacy & federated\nlearning", N_, ""), ("Quantum-ML\n& neuromorphic", N_, "")]),
]
fig, ax = plt.subplots(figsize=(17, 14)); fig.patch.set_facecolor(SURF)
ax.set_xlim(-0.5, 100.5); ax.set_ylim(-0.5, 85); ax.axis("off")
for title, x, y, w, h, ncol, chips in D:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#ffffff", ec="#dddcd7", lw=1.2))
    ax.text(x + 1, y + h - 1.4, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    pad, top = 0.9, y + h - 4.2
    nrow = -(-len(chips) // ncol)
    ch = min(5.6, (h - 5 - pad * (nrow - 1)) / nrow)
    cw = (w - pad * (ncol + 1)) / ncol
    for i, (t, s, badge) in enumerate(chips):
        r, c = divmod(i, ncol)
        cx, cy = x + pad + c * (cw + pad), top - (r + 1) * ch - r * pad
        fc, ec, tc, ls = ST[s]
        ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=1.4, ls=ls))
        ax.text(cx + cw / 2 - (2.2 if badge else 0) * 0, cy + ch / 2 + (0.6 if badge else 0), t, ha="center", va="center", fontsize=10, color=tc, linespacing=1.15)
        if badge:
            ax.text(cx + cw / 2, cy + 0.75, badge, ha="center", va="bottom", fontsize=8.3, color=ec if s != "niet" else MUTED, fontweight="bold")
# titel + legenda
ax.text(0, 84, "De landkaart van AI en machine learning", fontsize=19, fontweight="bold", color=INK, va="top")
ax.text(0, 81.1, "Van fundament (onder) naar toepassing (boven). Engineering staat ernaast: dat heb je bij élke laag nodig.", fontsize=11.5, color=INK2, va="top")
lx = 0
for s, lab in [("had", "al gehad (vak, stage of thesis)"), ("bezig", "nu bezig / deels"), ("repo", "in deze cursus (M = module)"), ("niet", "niet in deze cursus")]:
    fc, ec, tc, ls = ST[s]
    ax.add_patch(FancyBboxPatch((lx, 76.4), 2.4, 1.8, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(lx + 3, 77.3, lab, fontsize=11, color=INK2, va="center"); lx += 3.4 + len(lab) * 0.62
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURF, pad_inches=0.3)
print("saved", OUT)
