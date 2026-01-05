import numpy as np


def run_episode(
    env, # PettingZoo environment
    agents, # List of agent
    ep_lenght_max = 25, # Maximum episode length
    seed = 0, # Random seed
    render = False # Render into terminal
):

    rng = np.random.default_rng(seed)

    n_agents = len(agents)
    if n_agents != env.n_agents: # Debug
        raise ValueError(f"Number of agents ({n_agents}) must match env ({env.n_agents})")

    actions_hist = np.zeros((ep_lenght_max, n_agents), dtype=float) # Actions history
    rewards_hist = np.zeros((ep_lenght_max, n_agents), dtype=float) # Rewards history
    p_hist = np.zeros((ep_lenght_max, n_agents), dtype=float) # Rho history
    A_hist = np.zeros((ep_lenght_max, n_agents), dtype=float) # Aspiration history

    # Must reset env and agent at start of episode
    obs, infos = env.reset(seed=seed)

    for agent in agents:
        agent.reset()

    for t in range(ep_lenght_max):
        # --- Agents choose actions (dict for PettingZoo) ---
        actions_dict = {}
        actions_vec = np.zeros(n_agents, dtype=float)



        for i, name in enumerate(env.possible_agents):
            a_arr = agents[i].select_action(rng) # np.ndarray shape (1,)
            a_val = float(np.asarray(a_arr, dtype=float).reshape(-1)[0])
            a_val = float(np.clip(a_val, 0.0, 1.0))



            actions_vec[i] = a_val
            actions_dict[name] = np.array([a_val], dtype=np.float32)

        # --- Env step ---
        obs, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

        # convert rewards dict into vector 
        rewards_vec = np.zeros(n_agents, dtype=float)
        for i, name in enumerate(env.possible_agents):
            rewards_vec[i] = float(rewards_dict.get(name, 0.0))

        # --- Update agents ---
        for i in range(n_agents):
            agents[i].update(rewards_vec[i])
            p_hist[t, i] = float(getattr(agents[i], "p", np.nan))
            A_hist[t, i] = agents[i].A


        # store history
        actions_hist[t] = actions_vec
        rewards_hist[t] = rewards_vec

        # --- Optional render ---
        if render and hasattr(env, "render"):
            env.render()

        # Time limit reached or terminated
        if any(truncations.values()) or any(terminations.values()):
            break

    return actions_hist, rewards_hist, p_hist, A_hist


def run_simulation(
    env, # PettingZoo environment
    agents, # agent list
    n_episodes = 1000, # number of independent runs
    ep_lenght_max = 25, # maximum episode length
    seed = 0, # random seed
    render = False # render first episode only into terminal
):
    
    n_agents = len(agents)
    actions_all = np.zeros((n_episodes, ep_lenght_max, n_agents), dtype=float) # actions
    rewards_all = np.zeros((n_episodes, ep_lenght_max, n_agents), dtype=float) # reward
    p_all = np.zeros((n_episodes, ep_lenght_max, n_agents), dtype=float) # Rhp
    A_all = np.zeros((n_episodes, ep_lenght_max, n_agents), dtype=float) # Aspiration

    for ep in range(n_episodes):
        # change seed each episode for reproducibility
        ep_seed = seed + ep
        
        actions_hist, rewards_hist, p_hist, A_hist = run_episode(
            env=env,
            agents=agents,
            ep_lenght_max=ep_lenght_max,
            seed=ep_seed,
            render = render if ep == 0 else False, # render only first episode by default
        )
        
        actions_all[ep] = actions_hist
        rewards_all[ep] = rewards_hist
        p_all[ep] = p_hist
        A_all[ep] = A_hist
        
    return actions_all, rewards_all, p_all, A_all
