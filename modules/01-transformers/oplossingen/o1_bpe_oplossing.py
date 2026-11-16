"""Oplossing O1 — BPE-tokenizer. Kijk pas als je zelf een poging hebt gedaan."""


def get_pair_counts(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {pair} -> {new_id} ({self.vocab[new_id]!r}), "
                      f"{counts[pair]}x")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # Paar met de laagste merge-index (eerst geleerd = eerst toepassen)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    tekst = (
        "De kat krabt de krullen van de trap. "
        "Een transformer voorspelt het volgende token, token na token. "
    ) * 20
    tok = BPETokenizer()
    tok.train(tekst, vocab_size=300)
    assert len(tok.merges) == 44
    for s in [tekst, "token", "Hallo wereld! 🚀", "ëéèê — Ünïcödé"]:
        assert tok.decode(tok.encode(s)) == s
    n_bytes, n_tokens = len(tekst.encode()), len(tok.encode(tekst))
    print(f"Compressie: {n_bytes} bytes -> {n_tokens} tokens ({n_bytes/n_tokens:.2f}x)")
    assert n_tokens < n_bytes / 2
    print("Alle tests geslaagd!")
