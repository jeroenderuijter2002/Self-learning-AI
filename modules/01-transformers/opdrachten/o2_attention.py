"""
Opdracht O2 — Attention from scratch in NumPy
=============================================

Geen PyTorch, geen magie: alleen matrixvermenigvuldiging en softmax.
Vul de TODO's in en draai:

    python o2_attention.py

Conventie voor shapes:
    T = aantal tokens (sequentielengte)
    d_model = embeddingdimensie
    d_k = dimensie per head (= d_model // n_heads)
"""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numeriek stabiele softmax.

    Hint: trek eerst het maximum af langs `axis` (waarom mag dat?).
    """
    # TODO
    raise NotImplementedError


def causal_mask(T: int) -> np.ndarray:
    """Bool-matrix (T, T): True waar token i naar token j MAG kijken (j <= i)."""
    # TODO
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Q, K, V: (T, d_k)
    mask:    (T, T) bool of None. Waar mask False is -> score = -inf.
    Returns: (output (T, d_k), weights (T, T))
    """
    # TODO
    raise NotImplementedError


def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, causal=True):
    """Multi-head self-attention.

    X:              (T, d_model)
    W_q, W_k, W_v:  (d_model, d_model)
    W_o:            (d_model, d_model)

    Stappen:
      1. Q = X @ W_q, idem K, V           -> (T, d_model)
      2. splits in n_heads stukken         -> (n_heads, T, d_k)
      3. attention per head
      4. plak de heads weer aan elkaar     -> (T, d_model)
      5. output-projectie met W_o
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # softmax
    s = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(s.sum(-1), 1.0)
    assert np.allclose(s[1], 1 / 3), "Softmax niet stabiel bij grote waarden"

    # mask
    m = causal_mask(3)
    assert m.tolist() == [[True, False, False],
                          [True, True, False],
                          [True, True, True]]

    # attention
    T, d_k = 5, 8
    Q, K, V = (rng.normal(size=(T, d_k)) for _ in range(3))
    out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
    assert out.shape == (T, d_k) and w.shape == (T, T)
    assert np.allclose(w.sum(-1), 1.0), "Elke rij gewichten moet optellen tot 1"
    assert np.allclose(np.triu(w, k=1), 0.0), "Toekomst lekt door de mask!"
    assert np.allclose(out[0], V[0]), "Token 0 kan alleen naar zichzelf kijken"

    # Experiment: zonder sqrt(d_k)-schaling wordt softmax 'piekerig'
    d_big = 512
    Qb, Kb = rng.normal(size=(T, d_big)), rng.normal(size=(T, d_big))
    raw = softmax(Qb @ Kb.T)
    scaled = softmax(Qb @ Kb.T / np.sqrt(d_big))
    print(f"Max gewicht zonder schaling: {raw.max(-1).mean():.3f} | "
          f"met schaling: {scaled.max(-1).mean():.3f}")

    # multi-head
    d_model, n_heads = 16, 4
    X = rng.normal(size=(T, d_model))
    Ws = [rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(4)]
    Y = multi_head_attention(X, *Ws, n_heads=n_heads)
    assert Y.shape == (T, d_model)

    # Causaliteit-check: veranderen van het LAATSTE token mag eerdere outputs
    # niet beïnvloeden.
    X2 = X.copy()
    X2[-1] += 10.0
    Y2 = multi_head_attention(X2, *Ws, n_heads=n_heads)
    assert np.allclose(Y[:-1], Y2[:-1]), "Eerdere tokens zien de toekomst!"

    print("Alle tests geslaagd!")
