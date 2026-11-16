# Module 5: Reinforcement learning

🟢 *Verbreding* · **Status: 🔜 gepland.** Deze pagina laat zien wat er komt; de notes, figuren, speeltuinen en opdrachten worden in dezelfde stijl als [module 1](../01-transformers/README.md) uitgewerkt.

## Waarom deze module?

RL is de techniek achter AlphaGo, robotica en (via module 2) RLHF en reasoning-modellen. Met je achtergrond in stochastische beslistheorie heb je de basis al.

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
