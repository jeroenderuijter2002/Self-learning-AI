# P1 · Praktijkopdracht: De Nederlandse token-belasting

*Je werkt als consultant. Een klant komt met een echte vraag; jij levert een onderbouwd antwoord. Dit is de afsluiter van module 1: je gebruikt wat je in 1.1 en O1 over tokenization hebt geleerd.*

## De klant

**Mediahuis Noord** (fictief; denk aan een grote Nederlandse uitgever). Ze laten elke dag ruim 2.000 artikelen samenvatten door een LLM-API.

## De vraag

> "Onze Engelstalige zustertitel doet hetzelfde met evenveel artikelen, maar hun AI-rekening is flink lager. Hoe kan dat, en wat kunnen we eraan doen?"

## Wat je krijgt

- Parallelle Nederlandse en Engelse teksten uit [Europarl](https://www.statmt.org/europarl/), en de tokenizers van GPT-2, GPT-4 en GPT-4o via `tiktoken`.
- Een prijs per miljoen tokens. Gebruik een parameter (bijv. `PRIJS_PER_M_TOKENS = 2.50`) en zoek de actuele prijs zelf op: API-prijzen veranderen vaak, en je advies moet blijven kloppen als de prijs verandert.

## Aanpak (richtlijn, geen keurslijf)

1. **Meet het verschil.** Neem een paar duizend parallelle zinnen uit Europarl (dezelfde inhoud in het Nederlands en het Engels). Tel per tokenizer (`gpt2`, `cl100k_base`, `o200k_base`) het aantal tokens. Hoeveel meer tokens kost Nederlands, per tokenizer?
2. **Zoek de oorzaak.** Welke Nederlandse woorden worden in de meeste stukjes geknipt? Kijk naar samenstellingen (*arbeidsongeschiktheidsverzekering*) en vervoegingen. 🔁 Weet je nog uit [1.1](../notes/1.1-tokens-en-embeddings.md) hoe BPE zijn vocabulaire leert, en op wat voor tekst?
3. **Reken het door.** Wat kost 2.000 artikelen per dag samenvatten per jaar, in het Nederlands tegenover het Engels? Maak een kleine gevoeligheidsanalyse: wat als de prijs halveert, of het aantal artikelen verdubbelt?
4. **Het addertje.** De directeur oppert: "Vertalen we alles toch eerst naar het Engels?" Reken uit of dat echt goedkoper is, en wat het kost aan kwaliteit en risico. Denk aan: de vertaalstap kost zelf ook tokens (in én uit), en vertalen kan nuances en namen beschadigen.
5. **Schrijf de memo.** Eén A4 voor de hoofdredacteur. Begin met het advies, dan de onderbouwing. Geen jargon: leg BPE uit met een vergelijking, niet met een formule.

## Wat je oplevert

Een notebook met de analyse en een memo van één pagina voor de hoofdredacteur (geen techneut) met je advies.

## Wanneer is het goed?

Je memo legt zonder jargon uit *waarom* Nederlands meer tokens kost (BPE, 1.1), onderbouwt het verschil met eigen metingen, en eindigt met een concreet, doorgerekend advies.

> 💡 **Tip:** laat Claude je memo lezen als "een hoofdredacteur zonder technische achtergrond" en vraag welke zin hij niet begrijpt.

---
[← Terug naar module 1](../README.md)
