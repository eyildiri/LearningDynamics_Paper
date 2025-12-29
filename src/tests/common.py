# tests/common.py
import numpy as np

def assert_allclose(actual, expected, tol=1e-12, msg=""):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if not np.all(np.abs(actual - expected) <= tol):
        idx = np.argmax(np.abs(actual - expected))
        raise AssertionError(
            f"{msg}\nMax error at index {idx}: actual={actual[idx]} expected={expected[idx]}\n"
            f"Actual min/max: {actual.min()} / {actual.max()}\n"
            f"Expected unique: {np.unique(expected)}"
        )

def rewards_dict_to_vec(env, rewards_dict):
    """Convert rewards dict (player_i) to a vector ordered by env.possible_agents."""
    return np.array([float(rewards_dict[name]) for name in env.possible_agents], dtype=float)

def neighbors_of(env, agent_idx):
    """Return neighbor indices for a given agent index (0..n_agents-1)."""
    return env.neighbors[agent_idx]
