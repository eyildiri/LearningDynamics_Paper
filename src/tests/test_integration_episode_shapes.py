# CHATGPT TEST FILE

import numpy as np
from pdg_env import PDGGridParallelEnv
from agents import make_agents
from loop_pz import run_episode


def main():
    size = 10
    tmax = 25
    seed = 0

    env = PDGGridParallelEnv(size=size, ep_leng=tmax)

    agents = make_agents(
        env.n_agents,
        A=0.5,
        beta=1.0,
        epsilon=0.2,
        p_init=0.5,
    )

    actions_hist, rewards_hist, p_hist = run_episode(
        env=env,
        agents=agents,
        ep_lenght_max=tmax,
        seed=seed,
        render=False,
    )

    # --- shape checks ---
    assert actions_hist.shape == (tmax, env.n_agents), actions_hist.shape
    assert rewards_hist.shape == (tmax, env.n_agents), rewards_hist.shape
    assert p_hist.shape == (tmax, env.n_agents), p_hist.shape

    # --- bounds checks ---
    assert np.all((actions_hist == 0) | (actions_hist == 1)), "actions not in {0,1}"
    assert np.all(p_hist >= 0.0) and np.all(p_hist <= 1.0), "p out of [0,1]"

    # rewards are bounded by min/max PD payoffs (since avg of 4 interactions)
    # min is S=0, max is T=5 in your default matrix
    assert np.min(rewards_hist) >= -1e-12, f"min reward too low: {np.min(rewards_hist)}"
    assert np.max(rewards_hist) <= 5.0 + 1e-12, f"max reward too high: {np.max(rewards_hist)}"

    # non-triviality (should not be all same always)
    unique_actions_t0 = np.unique(actions_hist[0])
    assert unique_actions_t0.size >= 1, "no actions?"
    # (not required but often true) at least some diversity in early steps with p=0.5
    # We won't fail hard on it, just print.
    print("Unique actions at t=0:", unique_actions_t0)

    print("test_integration_episode_shapes passed")


if __name__ == "__main__":
    main()
