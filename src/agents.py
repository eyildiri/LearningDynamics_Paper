import numpy as np
from BMmodel import C, D, apply_misimplementation, bm_step, fix01


class AspirationBMAgent:
    """
    Aspiration-based Bush-Mosteller (BM) learner.
        1) select_action(): uses a noisy version of p (misimplementation epsilon) to sample C/D
        2) update(reward): updates p using Eq.(2) stimulus + Eq.(1) BM update
    """

    def __init__(self, A, beta, epsilon, p_init=0.5):
        self.A = float(A)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.p = fix01(float(p_init)) 
        self.last_action = None        # we initialize it here to None because it's needed for the BM update (Eq. 1)

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
        action = C if rng.random() < p_eff else D   # pas tres bien sur si c'est juste !!!!!!!!!!!!!!!!
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


def make_agents(n_agents, A, beta, epsilon, p_init=0.5):
    """Create a list of identical AspirationBMAgents."""
    return [AspirationBMAgent(A=A, beta=beta, epsilon=epsilon, p_init=p_init) for _ in range(n_agents)]