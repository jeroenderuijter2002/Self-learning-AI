# Software-installatie

## Optie A: lokaal (aanbevolen)

```bash
# Python 3.10+ is nodig
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy torch matplotlib tiktoken transformers jupyter
```

Check of het werkt:

```bash
python -c "import torch, numpy, tiktoken; print('torch', torch.__version__)"
```

Een GPU is **niet** nodig voor module 1. Alles draait op een laptop-CPU, alleen O3 is dan wat trager. Op een Mac met Apple Silicon kan PyTorch de GPU gebruiken via `device="mps"`.

## Optie B: Google Colab

Ga naar [colab.research.google.com](https://colab.research.google.com), maak een notebook en zet bovenaan:

```python
!pip install tiktoken transformers
```

Zet een gratis GPU aan via *Runtime → Change runtime type → T4 GPU*. Dat is handig voor O3.

## Bestanden uit dit project gebruiken

De `.py`-bestanden staan in de projectdocumenten. Kopieer ze naar een lokale map, bijvoorbeeld `~/zelf-ai/`, met dezelfde mappenstructuur. Wil je er een echte git-repo van maken (aanrader, dan kun je je voortgang committen):

```bash
cd ~/zelf-ai && git init && git add . && git commit -m "start cursus"
```
