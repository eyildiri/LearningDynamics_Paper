from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


# PettingZoo ParallelEnv for PGG 
# - 4 players (default)
class PGGParallelEnv(ParallelEnv):

    metadata = {"name" : "pgg_parallel_env"}

    def __init__(
        self,
        n_agents : int = 4, # number of player
        ep_leng : int = 25, # episode length
        multiplication : float = 1.6, # public goods multiplier
        coop_threshold : float = 0.5 # X threshold to binarize contributions
    ):

        # --- Environment parameters ---
        # Variable attributions
        self.n_agents = n_agents
        self.ep_leng = ep_leng
        self.multiplication = multiplication
        self.coop_threshold = coop_threshold

        # PettingZoo agent names
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]
        self.agents: List[str] = []

        # Internal episode state
        self.ep_counter = 0
        self.rng = np.random.default_rng()
        self.last_actions: Optional[np.ndarray] = None # contributions in [0,1]

        # --- Spaces ---
        # Observation (agents are external and ignore obs => agents.py)
        self._obs_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

        # Action (continuous in [0,1])
        self._act_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


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
        seed = None, # random seed
        options = None, # additional options (unused)
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:

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
    # - ActionType = np.ndarray (dtype = float32) [0,1] (more robust than float for gymnasium)
    # - ObsType = np.ndarray
    # Returns : obs, rewards, terminations, truncations, infos
    def step(
        self,
        actions : Dict[str, np.ndarray] # agent_name -> action [0,1]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:

        if not self.agents: # check if env is done
            return {}, {}, {}, {}, {}

        contrib = self._parse_actions(actions) # np.ndarray shape (n_agents,)
        rewards_vec = self.compute_rewards(contrib)

        self.last_actions = contrib.copy()
        self.ep_counter += 1
        trunc = self.ep_counter >= self.ep_leng

        obs = {a: np.zeros((1,), dtype=np.float32) for a in self.agents}
        rewards = {a: float(rewards_vec[i]) for i, a in enumerate(self.agents)}
        terminations = {a : False for a in self.agents} # no terminal states
        truncations = {a : trunc for a in self.agents}
        infos = {a : {} for a in self.agents}

        # Add fraction of other players above threshold to infos
        fC = self.fraction_coop_others(self.last_actions)
        for i, a in enumerate(self.agents):
            infos[a]["fC"] = float(fC[i])

        if trunc:
            self.agents = []

        return obs, rewards, terminations, truncations, infos



    def _parse_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        contrib = np.zeros(self.n_agents, dtype=float)
        
        for i, name in enumerate(self.agents):
            raw = actions.get(name, 0.0)      
            arr = np.asarray(raw, dtype=float).reshape(-1)
            val = float(arr[0]) if arr.size else 0.0
            
            # Clamp to [0,1]
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0

            contrib[i] = val

        return contrib

    # Compute payoffs for the PGG
    # payoff_i = 1 - a_i + (multiplication / n_agents) * sum(a)
    def compute_rewards(
        self, 
        actions: np.ndarray
    ) -> np.ndarray:
        
        a = np.asarray(actions, dtype=float).reshape(self.n_agents,)
        total = float(np.sum(a))
        share = self.multiplication / self.n_agents
        
        return 1.0 - a + share * total


    # Compute fraction of cooperating player (above threshold) for all agent (for infos)
    # Do not use for plot because I'm not sure it's the exact condition => todo : verify
    def fraction_coop_others(
        self, 
        actions : np.ndarray # action of all agents
    ) -> np.ndarray:

        a = np.asarray(actions, dtype=float).reshape(self.n_agents,)
        coop = (a >= self.coop_threshold).astype(float)

        fC = np.zeros(self.n_agents, dtype=float)
        
        for i in range(self.n_agents):
            others = [j for j in range(self.n_agents) if j != i]
            fC[i] = float(np.mean(coop[others]))
        
        return fC


    # Print into the terminal the last action :
    # - player's contribution
    # - mean contribution and global fraction above threshold
    def render(self) -> None:

        print(f"=== EPISODE {self.ep_counter} ===")
        
        if self.last_actions is None:
            print("=> NO ACTION ")
            return

        a = self.last_actions
        for i, name in enumerate(self.agents):
            print(f"=> {name} : a = {a[i]}")

        mean_contrib = float(np.mean(a))
        coop_rate = float(np.mean(a >= self.coop_threshold))
        
        print(f"Mean contribution: {mean_contrib}")
        print(f"Cooperation rate (threshold : {self.coop_threshold}): {coop_rate}")


