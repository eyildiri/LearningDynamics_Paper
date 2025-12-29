# CHATGPT TEST FILE

import numpy as np
from pdg_env import PDGGridParallelEnv
from tests.common import assert_allclose, rewards_dict_to_vec


def main():
    env = PDGGridParallelEnv(size=10, ep_leng=1)
    env.reset(seed=0)

    actions = {name: 0 for name in env.possible_agents}  # all defect
    obs, rewards, terminations, truncations, infos = env.step(actions)

    r = rewards_dict_to_vec(env, rewards)

    # Expected: D vs D gives P=1 on each of 4 neighbors, average is still 1
    expected = np.full(env.n_agents, 1.0, dtype=float)
    assert_allclose(r, expected, msg="All-D test failed")

    print("test_all_d passed (all rewards = 1.0)")

if __name__ == "__main__":
    main()
