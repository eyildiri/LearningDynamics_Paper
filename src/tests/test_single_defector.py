# CHATGPT TEST FILE

import numpy as np
from pdg_env import PDGGridParallelEnv
from tests.common import assert_allclose, rewards_dict_to_vec, neighbors_of

def main():
    env = PDGGridParallelEnv(size=10, ep_leng=1)
    env.reset(seed=0)

    # Everyone cooperates except one defector at index 0
    defect_idx = 0
    actions = {name: 1 for name in env.possible_agents}
    actions[f"player_{defect_idx}"] = 0

    obs, rewards, terminations, truncations, infos = env.step(actions)
    r = rewards_dict_to_vec(env, rewards)

    # Expected:
    # - defector vs 4 cooperators => T=5 each => average 5
    # - each neighbor of defector: has 3 cooperative neighbors (R=3) and 1 defector neighbor (S=0)
    #   average = (3+3+3+0)/4 = 2.25
    # - everyone else still surrounded by cooperators => 3.0
    expected = np.full(env.n_agents, 3.0, dtype=float)
    expected[defect_idx] = 5.0

    neigh = neighbors_of(env, defect_idx)
    for j in neigh:
        expected[j] = 2.25

    assert_allclose(r, expected, msg="Single-defector test failed")

    print("test_single_defector passed")
    print(f"   defector reward: {r[defect_idx]}")
    print(f"   neighbors ({neigh}) rewards: {[r[j] for j in neigh]}")

if __name__ == "__main__":
    main()
