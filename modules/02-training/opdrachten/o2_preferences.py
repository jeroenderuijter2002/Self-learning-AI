"""
Opdracht O2: Een reward model en de DPO-loss
============================================

Deel A: vul de TODO's in en draai de tests (seconden, geen downloads):

    python o2_preferences.py

Deel B: train een echt reward model op Anthropic HH-RLHF (~15-25 min op een CPU,
downloadt SmolLM2-135M en een deel van de dataset):

    python o2_preferences.py --echt

Notatie: w = het gekozen antwoord (winner), l = het afgewezen antwoord (loser).
"""
import sys

import torch
import torch.nn.functional as F


def bradley_terry_prob(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """P(w wordt verkozen boven l) = sigmoid(r_w - r_l). (notes 2.4)"""
    # TODO
    raise NotImplementedError


def reward_model_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """Gemiddelde negatieve log-likelihood van de voorkeuren: -log sigmoid(r_w - r_l).

    Gebruik F.logsigmoid, niet torch.log(torch.sigmoid(...)). Waarom? (Probeer r_w - r_l = -100.)
    """
    # TODO
    raise NotImplementedError


def sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Som van log p(token) over de antwoord-tokens, per voorbeeld.

    logits:      (B, T, V)  uitvoer van het model voor input_ids
    input_ids:   (B, T)
    answer_mask: (B, T)     1 op posities die bij het antwoord horen, anders 0
    Returns:     (B,)

    Let op de verschuiving: logits[:, t] voorspelt input_ids[:, t+1].
    Dit is dezelfde bouwsteen als de SFT-loss uit notes 2.3, maar dan als som per voorbeeld.
    """
    # TODO
    raise NotImplementedError


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1):
    """De DPO-loss (notes 2.5).

    pi_w, pi_l:   (B,) log pi_theta(y_w|x), log pi_theta(y_l|x)  (uit sequence_logprob)
    ref_w, ref_l: (B,) dezelfde log-kansen onder het bevroren referentiemodel
    Returns: (loss: scalar, reward_w: (B,), reward_l: (B,))
             met reward = beta * (log pi - log pi_ref), de impliciete beloning.
             Geef de impliciete beloningen terug als .detach() (alleen om te loggen).
    """
    # TODO
    raise NotImplementedError


def train_reward_head(feat_w: torch.Tensor, feat_l: torch.Tensor, steps: int = 300,
                      lr: float = 1e-2, l2: float = 1e-2) -> torch.Tensor:
    """Train een lineaire reward-kop r(x) = feat(x) @ w op voorkeursparen.

    feat_w, feat_l: (n, d) features van de gekozen en afgewezen antwoorden.
    Gebruik Adam en reward_model_loss, plus l2 * ||w||^2 als regularisatie.
    Begin met w = 0. Returns: w met shape (d,).
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Deel B: een echt reward model op HH-RLHF
# ---------------------------------------------------------------------------
def run_real(n_train=2000, n_test=600, model_name="HuggingFaceTB/SmolLM2-135M"):
    """Features van een bevroren SmolLM2 + jouw lineaire kop. Vergelijk met 'kies het langste'."""
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name).eval()
    tok.padding_side = tok.truncation_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token

    @torch.no_grad()
    def feats(texts, bs=8):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok([t + tok.eos_token for t in texts[i:i + bs]], return_tensors="pt",
                      padding=True, truncation=True, max_length=256)
            out.append(lm(**enc, output_hidden_states=True).hidden_states[-1][:, -1])
            print(f"\r  features {i + bs}/{len(texts)}", end="", flush=True)
        print()
        return torch.cat(out)

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    tr = random.Random(0).sample(list(ds["train"]), n_train)
    te = random.Random(0).sample(list(ds["test"]), n_test)
    Fw, Fl = feats([r["chosen"] for r in tr]), feats([r["rejected"] for r in tr])
    Tw, Tl = feats([r["chosen"] for r in te]), feats([r["rejected"] for r in te])
    mu, sd = Fw.mean(0), Fw.std(0) + 1e-6
    w = train_reward_head((Fw - mu) / sd, (Fl - mu) / sd)
    acc = (((Tw - mu) / sd) @ w > ((Tl - mu) / sd) @ w).float().mean().item()
    last = lambda d: d.rsplit("\n\nAssistant:", 1)[-1]
    longer = [len(last(r["chosen"])) > len(last(r["rejected"])) for r in te
              if len(last(r["chosen"])) != len(last(r["rejected"]))]
    print(f"Reward model: {acc:.1%} van de testparen goed geordend")
    print(f"Baseline 'kies het langste antwoord': {sum(longer) / len(longer):.1%}")


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
def _tests():
    torch.manual_seed(0)
    # Bradley-Terry
    p = bradley_terry_prob(torch.tensor([2.0, 0.0]), torch.tensor([0.5, 0.0]))
    assert torch.allclose(p, torch.tensor([0.8176, 0.5]), atol=1e-4)
    # alleen het verschil telt
    assert torch.allclose(bradley_terry_prob(torch.tensor(12.0), torch.tensor(10.5)), p[0])

    # RM-loss: waarde, stabiliteit en gradiëntrichting
    rw, rl = torch.tensor([1.0, -0.5], requires_grad=True), torch.tensor([0.0, 0.5], requires_grad=True)
    loss = reward_model_loss(rw, rl)
    assert torch.isclose(loss, (-F.logsigmoid(torch.tensor(1.0)) - F.logsigmoid(torch.tensor(-1.0))) / 2)
    loss.backward()
    assert (rw.grad < 0).all() and (rl.grad > 0).all(), "gradient descent moet r_w omhoog en r_l omlaag duwen"
    big = reward_model_loss(torch.tensor([-100.0]), torch.tensor([0.0]))
    assert torch.isfinite(big) and abs(big.item() - 100.0) < 1e-3, "gebruik F.logsigmoid voor stabiliteit"

    # sequence_logprob: vergelijk met een handmatige berekening
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    ids = torch.randint(V, (B, T))
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    lp = torch.log_softmax(logits, -1)
    manual = torch.stack([lp[0, 1, ids[0, 2]] + lp[0, 2, ids[0, 3]] + lp[0, 3, ids[0, 4]],
                          lp[1, 0, ids[1, 1]] + lp[1, 1, ids[1, 2]]])
    assert torch.allclose(sequence_logprob(logits, ids, mask), manual, atol=1e-5), "let op de verschuiving t -> t+1"

    # DPO: bij de start (pi = ref) is de loss ln 2 en zijn de beloningen 0
    pw, pl = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
    loss, rw_, rl_ = dpo_loss(pw, pl, pw.clone(), pl.clone(), beta=0.1)
    assert abs(loss.item() - 0.6931) < 1e-4 and torch.allclose(rw_, torch.zeros(2))
    # als het model het gekozen antwoord waarschijnlijker maakt, daalt de loss
    loss2, rw2, rl2 = dpo_loss(pw + 5, pl, pw, pl, beta=0.1)
    assert loss2 < loss and torch.allclose(rw2, torch.tensor([0.5, 0.5]))
    # gradiënt: omhoog voor w, omlaag voor l
    a, b = pw.clone().requires_grad_(), pl.clone().requires_grad_()
    dpo_loss(a, b, pw, pl, beta=0.1)[0].backward()
    assert (a.grad < 0).all() and (b.grad > 0).all()
    # DPO-gewicht: een paar dat al goed staat draagt minder bij dan een paar dat fout staat
    a = torch.tensor([0.0, 0.0], requires_grad=True)
    dpo_loss(a, torch.zeros(2), torch.tensor([-50.0, 50.0]), torch.zeros(2), beta=0.1)[0].backward()
    assert abs(a.grad[0]) < abs(a.grad[1]), "paren die al goed staan moeten minder gewicht krijgen"

    # reward-kop op synthetische data met een 'echte' voorkeursrichting
    d, n = 16, 800
    w_true = torch.randn(d)
    Xa, Xb = torch.randn(n, d), torch.randn(n, d)
    a_wins = (Xa @ w_true + 0.5 * torch.randn(n)) > (Xb @ w_true)
    Fw = torch.where(a_wins[:, None], Xa, Xb); Fl = torch.where(a_wins[:, None], Xb, Xa)
    w = train_reward_head(Fw[:600], Fl[:600])
    assert w.shape == (d,)
    acc = ((Fw[600:] @ w) > (Fl[600:] @ w)).float().mean().item()
    assert acc > 0.85, f"test-nauwkeurigheid {acc:.2f} is te laag"
    print(f"Reward-kop op synthetische data: {acc:.0%} van de testparen goed")
    print("Alle tests geslaagd!")


if __name__ == "__main__":
    if "--echt" in sys.argv:
        run_real()
    else:
        _tests()
