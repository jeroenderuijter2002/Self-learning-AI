"""
Opdracht O1 — Bouw je eigen BPE-tokenizer
=========================================

Vul de functies met `TODO` in. Draai daarna dit bestand:

    python o1_bpe.py

Als alle asserts slagen, zie je "Alle tests geslaagd!".
Oplossing: ../oplossingen/o1_bpe_oplossing.py (pas kijken als je echt vastzit).
"""


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Tel hoe vaak elk opeenvolgend paar voorkomt.

    Voorbeeld: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    # TODO
    raise NotImplementedError


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Vervang elk voorkomen van `pair` in `ids` door `new_id`.

    Voorbeeld: merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    """
    # TODO
    raise NotImplementedError


class BPETokenizer:
    def __init__(self):
        # Start: elke byte (0..255) is een token.
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Leer (vocab_size - 256) merges uit `text`.

        Stappen:
          1. text -> UTF-8 bytes -> lijst van ints
          2. herhaal: vind het meest voorkomende paar, merge het tot een nieuw id
          3. sla de merge op in self.merges en het nieuwe token in self.vocab
        """
        assert vocab_size >= 256
        # TODO
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Zet tekst om naar token-ids met de geleerde merges.

        Hint: pas merges toe in de volgorde waarin ze geleerd zijn
        (laagste new_id eerst). Zoek steeds het paar in je huidige ids
        dat de laagste merge-index heeft.
        """
        # TODO
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Zet token-ids terug naar tekst."""
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests — niet aanpassen
# ---------------------------------------------------------------------------
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
        assert tok.decode(tok.encode(s)) == s, f"Round-trip faalt voor: {s!r}"

    n_bytes = len(tekst.encode("utf-8"))
    n_tokens = len(tok.encode(tekst))
    print(f"Compressie: {n_bytes} bytes -> {n_tokens} tokens "
          f"({n_bytes / n_tokens:.2f}x)")
    assert n_tokens < n_bytes / 2, "Je tokenizer comprimeert te weinig"

    print("Alle tests geslaagd!")
