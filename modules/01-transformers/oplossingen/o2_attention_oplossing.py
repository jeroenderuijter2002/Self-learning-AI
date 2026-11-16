"""Oplossing O2 — attention in NumPy."""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # verandert de uitkomst niet, voorkomt overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)  # (..., T, T)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ V, w


def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, causal=True):
    T, d_model = X.shape
    d_k = d_model // n_heads

    def split(M):  # (T, d_model) -> (n_heads, T, d_k)
        return M.reshape(T, n_heads, d_k).transpose(1, 0, 2)

    Q, K, V = split(X @ W_q), split(X @ W_k), split(X @ W_v)
    mask = causal_mask(T) if causal else None
    out, _ = scaled_dot_product_attention(Q, K, V, mask)  # (n_heads, T, d_k)
    out = out.transpose(1, 0, 2).reshape(T, d_model)       # heads aan elkaar
    return out @ W_o


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    s = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(s.sum(-1), 1.0) and np.allclose(s[1], 1 / 3)
    assert causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    T, d_k = 5, 8
    Q, K, V = (rng.normal(size=(T, d_k)) for _ in range(3))
    out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))
    assert np.allclose(w.sum(-1), 1.0) and np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(out[0], V[0])
    d_big = 512
    Qb, Kb = rng.normal(size=(T, d_big)), rng.normal(size=(T, d_big))
    print(f"Max gewicht zonder schaling: {softmax(Qb @ Kb.T).max(-1).mean():.3f} | "
          f"met schaling: {softmax(Qb @ Kb.T / np.sqrt(d_big)).max(-1).mean():.3f}")
    d_model, n_heads = 16, 4
    X = rng.normal(size=(T, d_model))
    Ws = [rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(4)]
    Y = multi_head_attention(X, *Ws, n_heads=n_heads)
    X2 = X.copy(); X2[-1] += 10.0
    assert np.allclose(Y[:-1], multi_head_attention(X2, *Ws, n_heads=n_heads)[:-1])
    print("Alle tests geslaagd!")
