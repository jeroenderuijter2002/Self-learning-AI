# Prompt: een module uitwerken in een nieuwe chat

Kopieer alles tussen de lijnen naar een nieuwe chat. Geef die chat ook toegang tot de repomap. Pas alleen het blok **"Deze keer"** aan per module.

---

Je gaat een module uitwerken van mijn zelfstudie-cursus **"Zelf AI aanleren"**. Ik wil dat je die volledig, zorgvuldig en in dezelfde kwaliteit als module 1 bouwt. Neem er de tijd voor.

## Deze keer

**Module 2: Van tekstvoorspeller naar assistent** (`modules/02-training/`).

De kernvraag van de module: GPT-2 en Claude hebben grofweg dezelfde architectuur. Waarom is het ene een tekstvoorspeller en het andere een behulpzame assistent die nadenkt? Het antwoord zit in de training. De geplande hoofdstukken staan op de huidige module-pagina:
1. pretraining;
2. scaling laws;
3. supervised fine-tuning;
4. RLHF;
5. Constitutional AI en DPO;
6. reasoning-modellen.

Module 1 eindigt met een cliffhanger hierover; sluit daarop aan.

Ideeën die ik graag terugzie (alleen doen als het goed past):
- een scaling-curve met mijn eigen mini-GPT uit O3 van module 1;
- een klein reward model trainen op voorkeursparen;
- een klein open model finetunen met LoRA;
- een speeltuin waarin je ziet hoe een basismodel en een instruct-model hetzelfde prompt verschillend afmaken;
- een speeltuin met een scaling-law-calculator (Chinchilla: gegeven een compute-budget, wat is de optimale modelgrootte?).

## Wie ik ben en wat ik wil

- Ik ben Jeroen, MSc-student AI & Engineering Systems aan de TU/e. Ik wil AI/ML engineer of AI-consultant worden, en ik wil echt snappen hoe moderne AI zoals Claude werkt.
- **Ik heb al gehad:**
  - wiskunde en statistiek, optimalisatie;
  - de ML-basis (regressie, classificatie, SVM, clustering);
  - neurale netwerken en CNN's, Bayesiaanse ML en GP's;
  - OR, MDP's, regeltechniek en MPC;
  - evolutionaire algoritmen.
  - RL heb ik deels gehad.

  **Leg die basis niet opnieuw uit.** Verwijs er kort naar.
- Ik leer het best van **uitleg in verhaalvorm, zoals 3Blue1Brown**: eerst intuïtie en een concreet voorbeeld, dan pas de formule. Gebruik "🔁 Weet je nog?"-verwijzingen naar eerdere stof (met links), figuren met **echte data**, en dingen die ik zelf kan uitproberen.
- Ik wil een cursus zoals [bmlip/course](https://github.com/bmlip/course): een repo met notes, opdrachten, video's en verwijzingen naar goede bronnen.

## De repo

- **Lokaal:** `/Users/jeroenderuijter/Documents/Zelfstudie/AI/Self-learning-AI`. Je hebt toegang tot deze map; schrijf je bestanden direct daarin.
- **GitHub (publiek):** https://github.com/jeroenderuijter2002/Self-learning-AI
- **Speeltuinen via GitHub Pages:** https://jeroenderuijter2002.github.io/Self-learning-AI/interactief/
- **Ik commit en push zelf.** Doe geen git-commits; geef me aan het eind de commando's.
- Er is ook een claude.ai-Project "Zelf AI aanleren". Als je daar toegang toe hebt, zet je de belangrijkste nieuwe markdownbestanden er ook in, onder hetzelfde pad als in de repo.

## Lees dit eerst, vóór je iets schrijft

1. `STIJLGIDS.md`: **alle huisregels** (toon, opbouw, figuren-huisstijl, speeltuinen, opdrachten met tests, praktijkopdracht, links, wiskunde op GitHub, wat je na afloop bijwerkt). Volg die precies.
2. `README.md`: het overzicht, de moduletabel en het routediagram.
3. **Module 1 volledig**: `modules/01-transformers/` (README, alle notes, opdrachten en skeletons, zelftoets, `P1`), plus `tools/make_figures.py`, `tools/export_interactive_data.py` en een paar pagina's in `interactief/`. Dit is de kwaliteitslat en de stijl die ik wil.
4. De huidige pagina van de module die je gaat maken. Daar staan de motivatie, het 🌍-praktijkvoorbeeld, de geplande hoofdstukken, opdrachten, bronnen en de **💼 praktijkopdracht**. Dit is je startpunt: werk het uit en verbeter het waar nodig, maar gooi het niet weg.

## Werkwijze

1. **Eerst een plan, dan bouwen.** Stuur me na het lezen een kort plan:
   - de hoofdstukken (notes) met per hoofdstuk de kernvraag en het "aha-moment";
   - welke figuren met welke echte data;
   - welke speeltuin(en);
   - welke code-opdrachten;
   - hoe de praktijkopdracht wordt uitgewerkt;
   - wat er computationeel haalbaar is op een laptop of gratis Colab.

   Wacht op mijn akkoord. Antwoord ik niet, ga dan door met je beste plan.
2. **Bouw in stappen** en houd een takenlijst bij:
   1. eerst de figuren-scripts en data;
   2. dan de notes;
   3. dan de opdrachten met skeletons, tests en oplossingen;
   4. dan de speeltuin;
   5. dan de zelftoets en de praktijkopdracht;
   6. als laatste de module-README met leespad.
3. **Controleer alles echt:**
   - draai de figuren-scripts en bekijk elke figuur;
   - draai alle tests tegen de oplossingen;
   - test elke speeltuin in een headless browser;
   - check alle links;
   - zoek elk feit, jaartal en getal na en zet er een bron bij.

   Klopt iets niet zeker, formuleer het dan voorzichtiger of laat het weg.
4. **Werk na afloop bij** (zie het einde van `STIJLGIDS.md`):
   - de moduletabel in de hoofd-README (✅);
   - het routediagram (module 1 groen, 📍 naar de volgende module);
   - de speeltuinentabel en eventuele nieuwe mini's.
5. **Sluit af met** een korte samenvatting van wat er gemaakt is en de git-commando's om te pushen.

## Wat ik niet wil

- Telegramstijl of opsommingen in plaats van uitleg.
- Basis-ML opnieuw uitleggen.
- Verzonnen getallen, citaten of claims over modelgedrag die je niet hebt gecontroleerd.
- Figuren die alleen decoratie zijn, of waarvan labels overlappen.
- Een module die alleen leest: ik wil zelf bouwen, meten en uitproberen.

---
