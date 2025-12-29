# CHATGPT TEST FILE

import numpy as np
from pdg_env import PDGGridParallelEnv
from tests.common import assert_allclose, rewards_dict_to_vec


def main():
    env = PDGGridParallelEnv(size=10, ep_leng=1)
    env.reset(seed=0)

    actions = {name: 1 for name in env.possible_agents}  # all cooperate
    obs, rewards, terminations, truncations, infos = env.step(actions)

    r = rewards_dict_to_vec(env, rewards)

    # Expected: C vs C gives R=3 on each of 4 neighbors, average is still 3
    expected = np.full(env.n_agents, 3.0, dtype=float)
    assert_allclose(r, expected, msg="All-C test failed")

    print("test_all_c passed (all rewards = 3.0)")

if __name__ == "__main__":
    main()
