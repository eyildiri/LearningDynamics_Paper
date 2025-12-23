import numpy as np


class PDGEnv:
    """
    Prisoner's Dilemma Game on a 2D grid.
    - Grid: 10x10 agents
    - Each agent plays PDG with its 4 neighbors (up, down, left, right)
    - Each agent uses the SAME action against all neighbors in that round
    - Reward per agent is the AVERAGE payoff over the 4 neighbor interactions (as in the paper)
    """

    def __init__(self, size=10, R=3.0, T=5.0, S=0.0, P=1.0):
        self.size = int(size)
        self.n_agents = self.size * self.size

        # PDG payoffs 
        self.R = float(R)  # C,C
        self.T = float(T)  # D vs C
        self.S = float(S)  # C vs D
        self.P = float(P)  # D,D

        # Precompute neighbors for each agent index (0..n_agents-1)
        self.neighbors = self._build_neighbors()

    def reset(self, seed=None):
        if seed is not None:
            # Optional: set seed externally (loop will usually handle rng itself)
            np.random.seed(seed)

    def _idx(self, i, j):
        """Convert 2D coords (i,j) to 1D index."""
        return i * self.size + j

    def _build_neighbors(self):
        """
        Periodic boundary conditions:
        - up of row 0 is row size-1
        - down of row size-1 is row 0
        - left of col 0 is col size-1
        - right of col size-1 is col 0
        """
        neigh = []
        for _ in range(self.n_agents):
            neigh.append([])

        for i in range(self.size):
            for j in range(self.size):
                me = self._idx(i, j)

                up = self._idx((i - 1) % self.size, j)
                down = self._idx((i + 1) % self.size, j)
                left = self._idx(i, (j - 1) % self.size)
                right = self._idx(i, (j + 1) % self.size)

                neigh[me] = [up, down, left, right]

        return neigh

    def _pair_payoff(self, a_i, a_j):
        """
        Return payoff for player i when i plays a_i and neighbor plays a_j.
        """
        if a_i == 1 and a_j == 1:  # C,C
            return self.R
        if a_i == 0 and a_j == 1:  # D,C
            return self.T
        if a_i == 1 and a_j == 0:  # C,D
            return self.S
        if a_i == 0 and a_j == 0:  # D,D
            return self.P

    def step(self, actions):
        """
        Compute rewards for all agents given their actions.

        Args:
            actions: array-like of length n_agents with values 0 or 1

        Returns:
            rewards: np.ndarray shape (n_agents,), average payoff over 4 neighbors
        """
        actions = np.asarray(actions, dtype=int)
        rewards = np.zeros(self.n_agents, dtype=float)

        # For each agent, compute payoff vs each of its 4 neighbors, then average
        for i in range(self.n_agents):
            a_i = actions[i]
            total = 0.0
            for j in self.neighbors[i]:
                a_j = actions[j]
                total += self._pair_payoff(a_i, a_j)
            rewards[i] = total / 4.0

        return rewards
