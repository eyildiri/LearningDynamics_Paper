# loop.py
import numpy as np


def run_episode(env, agents, tmax=25, seed=0):
    """
    Run ONE episode of length tmax (like in the paper: tmax = 25 by default).

    Steps each time t:
      1) each agent selects an action (C/D)
      2) env computes rewards for all agents
      3) each agent updates its internal p using its reward

    Returns:
      actions_hist: np.ndarray shape (tmax, n_agents) with 0/1 actions
      rewards_hist: np.ndarray shape (tmax, n_agents) with float rewards
      p_hist: np.ndarray shape (tmax, n_agents) with each agent's intended p after update
    """
    rng = np.random.default_rng(seed)

    n_agents = len(agents)
    if n_agents != env.n_agents:
        raise ValueError(f"Number of agents ({n_agents}) must match env.n_agents ({env.n_agents})")

    actions_hist = np.zeros((tmax, n_agents), dtype=int)
    rewards_hist = np.zeros((tmax, n_agents), dtype=float)
    p_hist = np.zeros((tmax, n_agents), dtype=float)

    # (Optional) env reset for API consistency
    env.reset(seed=seed)

    for t in range(tmax):
        # 1) Agents choose actions
        actions = np.zeros(n_agents, dtype=int)
        for i, agent in enumerate(agents):
            actions[i] = agent.select_action(rng)

        # 2) Environment returns rewards (averaged over 4 neighbors)
        rewards = env.step(actions)

        # 3) Agents update their internal probability p using their own reward
        for i, agent in enumerate(agents):
            agent.update(rewards[i])
            p_hist[t, i] = agent.p

        # Store history
        actions_hist[t] = actions
        rewards_hist[t] = rewards

    return actions_hist, rewards_hist, p_hist


def run_simulation(env, agents, n_episodes=1, tmax=25, seed=0):
    """
    Run multiple episodes back-to-back.

    Returns:
      all_actions: shape (n_episodes, tmax, n_agents)
      all_rewards: shape (n_episodes, tmax, n_agents)
      all_p:       shape (n_episodes, tmax, n_agents)
    """
    n_agents = len(agents)

    all_actions = np.zeros((n_episodes, tmax, n_agents), dtype=int)
    all_rewards = np.zeros((n_episodes, tmax, n_agents), dtype=float)
    all_p = np.zeros((n_episodes, tmax, n_agents), dtype=float)

    for ep in range(n_episodes):
        # change seed each episode for reproducibility
        ep_seed = seed + ep

        actions_hist, rewards_hist, p_hist = run_episode(
            env=env,
            agents=agents,
            tmax=tmax,
            seed=ep_seed,
        )

        all_actions[ep] = actions_hist
        all_rewards[ep] = rewards_hist
        all_p[ep] = p_hist

    return all_actions, all_rewards, all_p
