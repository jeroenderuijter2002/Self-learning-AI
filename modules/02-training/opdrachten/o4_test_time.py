"""
Opdracht O4: Test-time compute en verifieerbare beloningen
==========================================================

Deel A: vul de TODO's in en draai de tests (seconden):

    python o4_test_time.py

Deel B: analyseer echte samples van Qwen2.5-0.5B-Instruct op GSM8K
(de metingen uit notes 2.6 staan in tools/results/m2_gsm8k.json):

    python o4_test_time.py --analyse

Deel C (optioneel, Colab-GPU aangeraden): genereer zelf samples, met je eigen prompt:

    python o4_test_time.py --genereer --n 20 --k 8
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def extract_answer(text: str):
    """Haal het eindantwoord uit een gedachtegang, als float (of None).

    1. Als er \\boxed{...} in staat: neem de inhoud van de LAATSTE \\boxed{}.
    2. Anders: neem het laatste getal in de tekst.
    Verwijder komma's als duizendtalscheiding ("1,000" -> 1000), een $-teken en een punt aan het eind.
    Lukt het niet om er een getal van te maken: None.
    """
    # TODO
    raise NotImplementedError


def majority_vote(answers):
    """Het antwoord dat het vaakst voorkomt (None-antwoorden tellen niet mee).
    Bij gelijkspel: het antwoord dat het eerst voorkwam. Alle antwoorden None: None."""
    # TODO
    raise NotImplementedError


def pass_at_k(n: int, c: int, k: int) -> float:
    """Zuivere schatter van pass@k (Chen et al., 2021, het Codex-paper):

        pass@k = 1 - C(n - c, k) / C(n, k)

    met n samples waarvan c goed. Dat is: 1 min de kans dat k willekeurig gekozen
    samples (zonder teruglegging) allemaal fout zijn. Gebruik math.comb.
    """
    # TODO
    raise NotImplementedError


def grpo_advantages(rewards, eps: float = 1e-6):
    """GRPO-voordelen voor één groep antwoorden op dezelfde vraag (notes 2.6):

        A_i = (r_i - mean(r)) / (std(r) + eps)

    Gebruik de populatie-standaardafwijking (np.std, ddof=0). Returns een np.array."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Deel B: analyse van echte samples
# ---------------------------------------------------------------------------
RESULTS = Path(__file__).resolve().parents[3] / "tools" / "results" / "m2_gsm8k.json"


def analyse(path=RESULTS):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    n = len(rows[0]["answers"])
    ok = lambda a, g: a is not None and abs(a - g) < 1e-6
    print(f"{len(rows)} vragen, {n} samples per vraag\n")
    print(f"greedy (1 poging):        {np.mean([ok(r['greedy'], r['gold']) for r in rows]):.1%}")
    for k in (1, 2, 4, 8, 16):
        if k > n:
            break
        pk = np.mean([pass_at_k(n, sum(ok(a, r["gold"]) for a in r["answers"]), k) for r in rows])
        mv = np.mean([ok(majority_vote(r["answers"][:k]), r["gold"]) for r in rows])
        print(f"k = {k:2d}:  pass@k = {pk:6.1%}   majority vote (eerste k) = {mv:6.1%}")
    # Hoeveel vragen geven bij GRPO geen leersignaal?
    zero = sum(np.allclose(grpo_advantages([float(ok(a, r["gold"])) for a in r["answers"]]), 0) for r in rows)
    print(f"\nVragen waarop alle samples even goed (of fout) zijn -> GRPO leert er niets van: {zero}/{len(rows)}")


# ---------------------------------------------------------------------------
# Deel C: zelf genereren
# ---------------------------------------------------------------------------
def genereer(n_vragen=20, k=8, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    import random, torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dev).eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i in random.Random(1).sample(range(len(ds)), n_vragen):
        q, gold = ds[i]["question"], float(ds[i]["answer"].split("####")[-1].replace(",", ""))
        chat = tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
            s = model.generate(ids.repeat(k, 1), max_new_tokens=320, do_sample=True, temperature=0.7, top_p=0.95,
                               pad_token_id=tok.eos_token_id)
        rows.append({"gold": gold, "greedy": extract_answer(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)),
                     "answers": [extract_answer(tok.decode(x[ids.shape[1]:], skip_special_tokens=True)) for x in s]})
        print(len(rows), gold, rows[-1]["answers"], flush=True)
    Path("o4_mijn_samples.json").write_text(json.dumps(rows))
    analyse("o4_mijn_samples.json")


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
def _tests():
    assert extract_answer("So the answer is \\boxed{42}.") == 42.0
    assert extract_answer("First \\boxed{3}, wait, actually \\boxed{5}") == 5.0, "neem de LAATSTE boxed"
    assert extract_answer("She pays $1,250 in total.") == 1250.0
    assert extract_answer("x = 2.5, so 7.") == 7.0
    assert extract_answer("\\boxed{\\$18}") in (18.0, None)
    assert extract_answer("I don't know") is None

    assert majority_vote([3.0, 5.0, 3.0, None, 5.0, 5.0]) == 5.0
    assert majority_vote([None, 2.0, 7.0]) == 2.0, "bij gelijkspel: de eerste"
    assert majority_vote([None, None]) is None

    # pass@k tegen brute force: alle deelverzamelingen van k uit n
    from itertools import combinations
    for n, c, k in [(5, 2, 1), (5, 2, 2), (6, 1, 3), (4, 0, 2), (4, 4, 1)]:
        samples = [1] * c + [0] * (n - c)
        subsets = list(combinations(samples, k))
        brute = sum(any(s) for s in subsets) / len(subsets)
        assert abs(pass_at_k(n, c, k) - brute) < 1e-12, (n, c, k)
    assert pass_at_k(16, 4, 1) == 0.25, "pass@1 is gewoon de fractie goed"

    A = grpo_advantages([1, 0, 0, 0])
    assert np.isclose(A.mean(), 0) and A[0] > 0 and np.allclose(A[1:], A[1])
    assert np.isclose(A[0], 0.75 / math.sqrt(0.1875), atol=1e-4)
    assert np.allclose(grpo_advantages([1, 1, 1]), 0), "alles goed: geen leersignaal"
    print("Alle tests geslaagd!")


if __name__ == "__main__":
    if "--analyse" in sys.argv:
        analyse()
    elif "--genereer" in sys.argv:
        a = sys.argv
        genereer(n_vragen=int(a[a.index("--n") + 1]) if "--n" in a else 20, k=int(a[a.index("--k") + 1]) if "--k" in a else 8)
    else:
        _tests()
