# CHATGPT TEST FILE

import time
import numpy as np

from pdg_env import PDGGridParallelEnv
from agents import make_agents


def main():
    size = 10
    tmax = 10
    seed = 0

    env = PDGGridParallelEnv(size=size, ep_leng=tmax)

    agents = make_agents(
        env.n_agents,
        A=0.5,
        beta=1.0,
        epsilon=0.2,
        p_init=0.5,
    )

    rng = np.random.default_rng(seed)

    obs, infos = env.reset(seed=seed)
    for ag in agents:
        ag.reset()

    print("\nInitial state (before any step):")
    env.render()

    for t in range(tmax):
        print(f"\n=== Step {t} ===")

        # Agents choose actions
        actions = {}
        for i, name in enumerate(env.possible_agents):
            actions[name] = agents[i].select_action(rng)

        # Environment step
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Agents update
        for i, name in enumerate(env.possible_agents):
            agents[i].update(rewards[name])

        # Render grid
        env.render()

        time.sleep(0.5)  # slow down so you can see evolution

        if any(truncations.values()):
            break


if __name__ == "__main__":
    main()
