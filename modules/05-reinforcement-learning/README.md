# Module 5: Reinforcement learning

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

RL is de techniek achter AlphaGo, robotica en (via module 2) RLHF en reasoning-modellen. Met je achtergrond in stochastische beslistheorie heb je de basis al.

## 🌍 In de praktijk

**AlphaGo tegen Lee Sedol, maart 2016.** Go werd gezien als het "onkraakbare" bordspel: er zijn meer mogelijke posities dan atomen in het heelal, dus brute rekenkracht werkt niet. Experts dachten dat een computer pas over een decennium zou winnen. Toch versloeg DeepMinds AlphaGo de 18-voudig wereldkampioen Lee Sedol [met 4–1 in Seoul](https://deepmind.google/en/research/alphago/), voor meer dan 200 miljoen kijkers. AlphaGo leerde door miljoenen keren tegen zichzelf te spelen: reinforcement learning. Beroemd werd **zet 37** in partij 2: een zet die geen mens zou spelen, en die geniaal bleek.

**In de financiële wereld:** J.P. Morgan begon in 2017 met LOXM, een systeem dat met [deep reinforcement learning](https://marketsmedia.com/j-p-morgan-ai-trading-global) leerde om grote aandelenorders zo snel en goedkoop mogelijk uit te voeren, getraind op miljarden historische en gesimuleerde trades.

**En dichterbij:** je thesis bij ASML gebruikt RL om wafer-handler-architecturen te evalueren, geïnspireerd door AlphaChip. Na deze module snap je die methoden van binnen.

## Wat je al kent (en wat we dus overslaan)

> ⏭️ MDP's, Bellman-vergelijkingen, value iteration en policy iteration: die ken je uit stochastische beslistheorie. We starten waar de modelvrije methoden beginnen. Waar het nodig is, verwijzen de notes kort terug ("🔁 Weet je nog?") in plaats van het opnieuw uit te leggen.

## Hoofdstukken (gepland)

| Hoofdstuk | Inhoud |
|---|---|
| 5.1 Van MDP naar leren zonder model | Monte Carlo en TD-learning, Q-learning, exploratie tegenover exploitatie. |
| 5.2 Deep RL | DQN: Q-learning met een neuraal netwerk, replay buffers, target networks. |
| 5.3 Policy gradients | REINFORCE, actor-critic, PPO, en de link met RLHF uit module 2. |
| 5.4 Plannen en zoeken | MCTS en AlphaZero: leren én vooruitdenken. |

## Opdrachten en projecten (gepland)

- **Opdracht:** Q-learning en DQN op Gymnasium-omgevingen (CartPole, LunarLander).
- **Opdracht:** PPO van scratch, daarna vergelijken met een bibliotheekimplementatie.

## Belangrijkste bronnen

- 📖 Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (gratis)
- 🎓 OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com) · [Hugging Face Deep RL-cursus](https://huggingface.co/learn)
- 📄 [DQN](https://arxiv.org/abs/1312.5602) · [PPO](https://arxiv.org/abs/1707.06347)

---
[← Terug naar het overzicht](../../README.md)
