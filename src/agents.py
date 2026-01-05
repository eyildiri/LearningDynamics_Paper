import numpy as np
from BMmodel import *


class AspirationBMAgent:
    """
    Aspiration-based Bush-Mosteller (BM) learner.
        1) select_action() : uses a noisy version of p (misimplementation epsilon) to sample C/D
        2) update(reward) : updates p using Eq.(2) stimulus + Eq.(1) BM update
    """

    def __init__(self, A, beta, epsilon, p_init=0.5, eta : float = 0.0):
        self.A_init = float(A)
        self.A = self.A_init
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.p_init = float(p_init)
        self.eta = eta

        self.p = fix01(float(p_init)) 
        self.last_action = None # we initialize it here to None because it's needed for the BM update (Eq. 1)

    def get_effective_p(self):
        """
        Return the probability actually used to sample actions.
        This includes action-flip noise (epsilon).
        """
        return apply_misimplementation(self.p, self.epsilon)

    def select_action(self, rng):
        """
        Sample an action using effective probability.
        """
        p_eff = self.get_effective_p()
        action = C if rng.random() < p_eff else D
        self.last_action = action
        return action

    def update(self, reward):
        """
        Update p after receiving a reward.
        """
        if self.last_action is None:
            raise RuntimeError("update() called before select_action().")

        self.p = bm_step(
            p=self.p,
            action_prev=self.last_action,
            reward_prev=float(reward),
            A=self.A,
            beta=self.beta,
        )
        
        if self.eta > 0:
            self.A = (1 - self.eta) * self.A + self.eta * reward

    def reset(self):
        self.p = fix01(float(self.p_init))
        self.last_action = None
        self.A = self.A_init


def make_agents(n_agents, A, beta, epsilon, p_init=0.5):
    """Create a list of identical AspirationBMAgents."""
    return [AspirationBMAgent(A=A, beta=beta, epsilon=epsilon, p_init=p_init) for _ in range(n_agents)]



# Agent class for PGG game (Bush-Mosteller)
class AspirationBMPGGAgent:

    def __init__(
        self,
        A : float, # niveau d'aspiration (translate or else osef)
        beta : float, # stimulus sensibility
        coop_threshold : float, # threshold to consider cooperation
        p_init : float = 0.5, # initial tendency to contribute
        sigma : float = 0.2, # stimulus
        max_redraws : int = 10000, # redraw max (to not loop forever but keep long enough)
        eta : float = 0.0 # Dynamic aspiration (0.0 = aspiration fix like in the paper)
    ):
    
        self.A = A
        self.A_init = A
        self.beta = beta
        self.coop_threshold = coop_threshold
        self.p_init = p_init
        self.sigma = sigma
        self.max_redraws = max_redraws
        self.eta = eta

        self.p = fix01(p_init)
        self.last_action = None


    # Draw from N(mean, sigma^2) and reject samples outside [0,1] until success
    def truncated_gaussian_01(
        self, 
        rng : np.random.Generator, # random number generator
        mean : float # mean
    ) -> float:

        mean = float(mean)
        
        for _ in range(self.max_redraws):
            x = float(rng.normal(loc=mean, scale=self.sigma))
            if 0.0 <= x <= 1.0:
                return x
        
        print(f"[WARNING] max draw reach ({self.max_redraws})")
        x = float(rng.normal(loc=mean, scale=self.sigma))
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x) # unsure [0,1]


    # Sample a continuous contribution in [0,1] as np.ndarray shape (1,)
    def select_action(
        self, 
        rng : np.random.Generator # random number generator
    ) -> np.ndarray:
    
        a = self.truncated_gaussian_01(rng, mean=self.p)
        self.last_action = a
        return np.array([a], dtype=np.float32)


    # Update p after receiving reward
    def update(self, reward : float):
        
        if self.last_action is None:
            raise RuntimeError("update called before select_action")

        s_prev = stimulus(r = reward, A = self.A, beta = self.beta)
        action_prev = C if self.last_action >= self.coop_threshold else D
        self.p = bm_update(p = self.p, action_prev = action_prev, s_prev = s_prev)
        
        if self.eta > 0:
            self.A = (1 - self.eta) * self.A + self.eta * reward


    # It reset all value
    def reset(self):
        self.p = fix01(self.p_init)
        self.last_action = None
        self.A = self.A_init



# Create a list of AspirationBMPGGAgent
def make_pgg_agents(
    n_agents : int, # number of agent
    A : float, # niveau d'aspiration (translate or else osef)
    beta : float, # stimulus sensibility
    coop_threshold : float, # threshold to consider cooperation
    p_init : float = 0.5, # initial tendency to contribute
    p_init_mode : str = "fixed", # "fixed" or random start
    seed = None, # random seed
    sigma : float = 0.2 # Ecart-type du bruit d’action (translate or else osef)
):

    rng = np.random.default_rng(seed)
    agents = []
    
    for _ in range(n_agents):
        
        if p_init_mode == "fixed":
            p0 = float(p_init)
        else: # random start
            p0 = float(rng.random())
        
        agents.append(AspirationBMPGGAgent(A=A, beta=beta, coop_threshold=coop_threshold, p_init=p0, sigma=sigma))

    return agents