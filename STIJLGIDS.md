# Stijlgids: hoe een module gemaakt wordt

Dit bestand is voor wie (mens of Claude) een nieuwe module uitwerkt. Module 1 ([`modules/01-transformers/`](modules/01-transformers/README.md)) is de referentie: als iets hier niet staat, doe het zoals daar.

## Voor wie is de cursus?

- **Jeroen**: MSc AI & Engineering Systems (TU/e). Hij wil AI/ML engineer of AI-consultant worden en snappen hoe moderne AI (zoals Claude) écht werkt.
- **Al gehad, dus niet opnieuw uitleggen:**
  - wiskunde en statistiek, optimalisatie;
  - de ML-basis (regressie, classificatie, SVM, clustering, PCA);
  - neurale netwerken en CNN's, Bayesiaanse ML en GP's;
  - OR, MDP's en dynamisch programmeren, regeltechniek en MPC;
  - evolutionaire algoritmen, verantwoorde AI.
  - RL, AutoML en recommenders: alleen kort gehad.
- Verwijs bij voorkennis kort terug in plaats van het opnieuw te behandelen. Elke module noemt bovenaan wat wordt overgeslagen (⏭️).

## Taal en toon

- **Nederlands.** Vaktermen blijven Engels als dat in het vak gebruikelijk is (attention, fine-tuning, embedding).
- **Verhalend, zoals 3Blue1Brown.**
  - Begin met een vraag of puzzel, bouw de intuïtie op, en geef pas daarna de formule.
  - Gebruik concrete voorbeelden met echte getallen.
  - Spreek de lezer aan met "je".
- **Geen telegramstijl.** Notes zijn doorlopend proza met tussenkopjes, geen opsommingen van steekwoorden.
- **🔁 Weet je nog?-blokken** verbinden nieuwe stof met eerdere hoofdstukken, modules of mini's. Gebruik daarbij altijd een link:
  `> 🔁 **Weet je nog?** ... (zie [Mini M3](../../../minis/M3-inproduct-en-matrices.md))`
- **▶ Probeer het zelf-blokken** verwijzen naar een speeltuin op het moment dat die de stof laat voelen.
- **Feiten kloppen.**
  - Elk jaartal, getal en citaat is nagezocht en heeft een bron-link.
  - Bij twijfel formuleer je voorzichtiger of laat je het weg.
  - Claims over modelgedrag ("head X doet Y") zijn gecontroleerd op de echte data.

## Opbouw van een module

```
modules/NN-naam/
  README.md          ← module-pagina (zie hieronder)
  notes/             ← N.1 … N.x, verhalende uitleg, in leesvolgorde
  figures/           ← PNG's, gemaakt door tools/make_figures.py (of een eigen script)
  opdrachten/        ← ON-*.md + oN_*.py skeletons met ingebouwde tests, en PN-*.md (praktijkopdracht)
  oplossingen/       ← volledige uitwerkingen van de code-opdrachten (getest!)
  zelftoets.md       ← vragen met <details>-antwoorden, gegroepeerd per hoofdstuk
```

**De module-README bevat:**
1. waarom de module ertoe doet;
2. 🌍 In de praktijk: een echt, nagezocht voorbeeld met bronnen;
3. leerdoelen als checklist;
4. voorkennis (⏭️);
5. "Wat je gaat zien": een tabel met de mooiste figuren;
6. het **leespad**: een tabel met stap, type, wat en tijd, waarin video's, notes, speeltuinen, opdrachten, toets en praktijkopdracht elkaar afwisselen. Typen: 🎬 Video · 📖 Notes · ▶ Speeltuin · 🛠️ Opdracht · ✅ Toets · 💼 Praktijk;
7. totale tijd;
8. externe speeltuinen en verder lezen.

De huidige "gepland"-pagina van elke module (motivatie, 🌍, hoofdstukken, opdrachten, 💼 praktijkopdracht, bronnen) is het startpunt. Werk die uit; gooi hem niet weg.

**Omvang:** een module is een hoofdstuk, geen collegeblok, maar wel volwaardig: ongeveer 15–25 uur studie, 4–6 notes, 8–15 figuren, 2–4 code-opdrachten, minstens één speeltuin, een zelftoets en de praktijkopdracht.

## Figuren

- **Echte data waar het kan:** echte modeloutputs, echte metingen, echte datasets. Een illustratie mag als schema, maar zeg dat dan.
- **Reproduceerbaar.** Elk figuur wordt gemaakt door een script in `tools/`. Breid `tools/make_figures.py` uit, of maak `tools/make_figures_moduleNN.py`. Gedownloade data komt in `tools/data/`, en die map staat in `.gitignore`.
- **Huisstijl matplotlib:**
  - achtergrond `#fcfcfb`;
  - inkt `#0b0b0b`, secundaire tekst `#52514e`, gedempt `#8a8984`, grid `#e6e5e1`;
  - categorische kleuren in deze vaste volgorde: blauw `#2a78d6`, oranje `#eb6834`, aqua `#1baf7a`;
  - sequentiële kleurenschaal van `#fcfcfb` naar `#0d366b` (blauw);
  - dunne lijnen, een rustig grid, geen dubbele y-as.
- **Bijschriften staan in de markdown, niet in de PNG.** De titel in de figuur is kort, de uitleg staat eronder in cursief.
- **Bekijk elke figuur.** Labels mogen niet overlappen of buiten beeld vallen.

## Speeltuinen (`interactief/`)

- Losse HTML-pagina's die `style.css` en `common.js` delen. Ze werken zonder server: ook met een dubbelklik, en via GitHub Pages op `https://jeroenderuijter2002.github.io/Self-learning-AI/interactief/`.
- Zware data (echte modeloutputs) wordt vooraf geëxporteerd naar `interactief/data/*.js` als `window.X = {...}`, door een script in `tools/`.
- Zet elke nieuwe speeltuin in `interactief/index.html` en in de speeltuinentabel in de hoofd-README.
- **Test in een headless browser:** geen console-errors, en elke knop of slider doet iets.

## Code-opdrachten

- Een skeleton `oN_naam.py` met `# TODO`-functies, docstrings, en onderaan tests die "Alle tests geslaagd!" printen.
- De oplossing in `oplossingen/` haalt alle tests. Draai ze echt.
- Gebruik Python, NumPy en PyTorch. Een opdracht moet op een laptop of gratis Colab kunnen draaien; noem het als er een GPU nodig is.

## 💼 Praktijkopdracht

Elke module eindigt met een consultancy-case: een (fictieve) klant met een echte vraag, publieke data, een concreet product (memo, slides, notebook, demo), een addertje en criteria voor "goed". De case staat al op de module-pagina. Bij het uitwerken komt er een eigen bestand `opdrachten/PN-*.md` met een aanpak in stappen en verwijzingen naar de notes (zie [P1](modules/01-transformers/opdrachten/P1-token-belasting.md)). Geef geen uitwerking: dit is het eigen werk.

## Links en techniek

- **Relatieve links:**
  - vanuit `modules/NN/` naar de mini's: `../../minis/...`;
  - vanuit `modules/NN/notes/` of `opdrachten/`: `../../../minis/...`;
  - naar andere modules vanuit `notes/`: `../../NN-naam/README.md`.
- **Wiskunde op GitHub:**
  - gebruik `$...$` en `$$...$$`;
  - gebruik geen `\,` (wordt een komma);
  - begin geen regel met `$$` direct na een lijstitem;
  - controleer de weergave.
- **Controleer na afloop alle links:** relatieve paden moeten bestaan, en externe links moeten bereikbaar zijn.

## Na het uitwerken van een module

1. **Hoofd-README, moduletabel:** de status wordt ✅ beschikbaar en de modulenaam wordt vet.
2. **Hoofd-README, routediagram (Mermaid):**
   - de vorige module krijgt `class ... had` (groen);
   - verplaats `📍 je bent hier` naar de volgende module en geef die `class ... nu`;
   - controleer dat het diagram rendert.
3. **Mini's:** heb je een concept dat steeds terugkomt? Maak er een mini van (`minis/M5-...md`) en zet hem in de tabel.
4. **Git:** commit of push niets. Dat doet Jeroen zelf. Geef aan het eind de `git add`/`commit`/`push`-commando's.
