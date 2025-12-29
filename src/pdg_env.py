from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from dataclasses import dataclass


# Data class for PDG payoffs (float value because computation bug)
@dataclass(frozen=True)
class PDGPayoffs:
    R : float = 3.0
    T : float = 5.0
    S : float = 0.0
    P : float = 1.0


# PettingZoo ParallelEnv for PDG on 2D grid
# - 4 neighbors
# - One action per agent per round for all neighbors (reward is average payoff)
class PDGGridParallelEnv(ParallelEnv):

    metadata = {"name": "pdg_parallel_env"}

    def __init__(
        self,
        size : int = 10, # grid size
        ep_leng : int = 25, # episode length
        payoffs : PDGPayoffs = PDGPayoffs(), # PDG payoffs
    ):
        
        # --- Environment parameters ---
        # Variable attributions
        self.size = size
        self.n_agents = size * size
        self.ep_leng = ep_leng
        self.payoffs = payoffs

        # PettingZoo agent names
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]
        self.agents : List[str] = []

        # Precompute neighbors for each agent
        self.neighbors : List[List[int]] = self._build_neighbors()

        # Internal episode state
        self.ep_counter = 0
        self.rng = np.random.default_rng()
        self.last_actions : Optional[np.ndarray] = None

        # --- Spaces ---
        # Observation (agents are external and ignore obs)
        self._obs_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        # Action (0 = D, 1 = C)
        self._act_space = spaces.Discrete(2)


    # Useless but required by PettingZoo
    def observation_space(self, agent: str):
        return self._obs_space


    # Useless but required by PettingZoo
    def action_space(self, agent: str):
        return self._act_space

    
    # PettingZoo reset method
    # - AgentID = str (player_*)
    # - ObsType = np.ndarray
    # Returns : obs, infos
    def reset(
        self, 
        seed : Optional[int] = None, # random seed
        options = None # additional options (unused)
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]] :

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:] # all agents return at reset as active
        self.ep_counter = 0
        self.last_actions = None

        obs = {a: np.zeros((1,), dtype=np.float32) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return obs, infos


    # PettingZoo step method
    # - AgentID = str (player_*)
    # - ActionType = int (0 or 1)
    # - ObsType = np.ndarray
    # Returns : obs, rewards, terminations, truncations, infos
    def step(
        self, 
        actions: Dict[str, int] # agent_name -> action (0 or 1)
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], 
            Dict[str, bool], Dict[str, bool], Dict[str, dict]]:

        if not self.agents: # check if env is done
            return {}, {}, {}, {}, {}

        # Convert agent name to array in order
        agent_vec = np.zeros(self.n_agents, dtype=int)
        for i, name in enumerate(self.possible_agents):
            agent_vec[i] = int(actions.get(name, 0)) # default action value = 0 (D)

        rewards_vec = self.compute_rewards(agent_vec)
        self.last_actions = agent_vec.copy()

        self.ep_counter += 1
        trunc = self.ep_counter >= self.ep_leng

        obs = {a: np.zeros((1,), dtype=np.float32) for a in self.agents}
        rewards = {a: float(rewards_vec[i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: False for a in self.agents} # No terminal states in PDG
        truncations = {a: trunc for a in self.agents} # Stop at ep_leng
        infos = {a: {} for a in self.agents}

        # Add fraction of cooperating neighbors to infos (for fig)
        if self.last_actions is not None:
            fC = self.fraction_coop_neighbors(self.last_actions) 
            for i, a in enumerate(self.possible_agents):
                infos[a]["fC"] = float(fC[i]) # fraction of cooperating neighbors

        if trunc: # if episode is over -> clear agents list
            self.agents = []

        return obs, rewards, terminations, truncations, infos


    # Print into the terminal the last action grid (C = 1, D = 0)
    def render(self) -> None:

        if self.last_actions is None:
            grid = np.full((self.size, self.size), -1, dtype=int)
        else:
            grid = self.last_actions.reshape(self.size, self.size)

        lines = []
        for i in range(self.size):
            row_syms = []
            for j in range(self.size):
                v = grid[i, j]
                if v == 1:
                    row_syms.append("C")
                elif v == 0:
                    row_syms.append("D")
                else:
                    row_syms.append(".")
    
            lines.append(" ".join(row_syms))

        print("\n".join(lines))
        
        coop_rate = grid.mean()
        print(f"\nCooperation rate: {coop_rate}\n")


    def _idx(self, i : int, j : int) -> int:
        """Convert 2D coords (i,j) to 1D index."""
        return i * self.size + j

    def _build_neighbors(self) -> List[List[int]]:
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


    # Payoff for player when player plays a_i and neighbor plays a_j
    def pair_payoff(
        self, 
        a_i : int, # action of player i
        a_j: int # action of neighbor j
    ) -> float:

        # 0 = D, 1 = C
        if a_i == 1 and a_j == 1:
            return self.payoffs.R # CC
        elif a_i == 0 and a_j == 1:
            return self.payoffs.T # CD
        elif a_i == 1 and a_j == 0:
            return self.payoffs.S # DC
        else :
            return self.payoffs.P # DD


    # Compute rewards for all agents given their actions
    def compute_rewards(
        self, 
        actions : np.ndarray # action for all active agents
    ) -> np.ndarray:

        rewards = np.zeros(self.n_agents, dtype=float)
        for i in range(self.n_agents):
            payoff_sum = 0.0
            a_i = actions[i]
            
            for j in self.neighbors[i]:
                payoff_sum += self.pair_payoff(a_i, actions[j])
                
            rewards[i] = payoff_sum / 4.0  # average over 4 neighbors

        return rewards


    # Compute fraction of cooperating neighbors for all agents (for infos)
    def fraction_coop_neighbors(self, actions: np.ndarray) -> np.ndarray:
        fC = np.zeros(self.n_agents, dtype=float)
        
        for i in range(self.n_agents):
            neigh = self.neighbors[i]
            fC[i] = float(np.mean(actions[neigh] == 1))

        return fC
