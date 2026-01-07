import math

# Action encoding
C = 1  # cooperate
D = 0  # defect


def fix01(x: float):
    """fix a value to [0, 1] to prevent calculation errors."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def stimulus(r: float, A: float, beta: float):
    """
    Eq. 2 from the paper:  s = tanh(beta * (r - A))
    """
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    return math.tanh(beta * (r - A))


def apply_misimplementation(p: float, epsilon: float):
    """
    Return the actual probability of cooperating after action-flip noise.
    Misimplementation (noise) from the paper: p_eps = p*(1-eps) + (1-p)*eps
    """
    p = fix01(p)
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")
    p_eps = p * (1.0 - epsilon) + (1.0 - p) * epsilon
    return fix01(p_eps)


def bm_update(p: float, action_prev: int, s_prev: float):
    """
    Eq. (1) from the paper (Bush-Mosteller update) : 
      If action_prev == C and s_prev >= 0: p <- p + (1-p)*s
      If action_prev == C and s_prev <  0: p <- p + p*s
      If action_prev == D and s_prev >= 0: p <- p - p*s
      If action_prev == D and s_prev <  0: p <- p - (1-p)*s
    """
    p = fix01(p)

    if action_prev not in (C, D):
        raise ValueError("action_prev must be 1 (C) or 0 (D)")

    # s_prev should be in (-1,1); we still accept edge cases and clamp lightly if needed.
    if s_prev <= -1.0:
        s_prev = -0.999999999
    elif s_prev >= 1.0:
        s_prev = 0.999999999

    if action_prev == C:
        if s_prev >= 0.0:
            # cooperated and satisfied -> increase p toward 1
            p_new = p + (1.0 - p) * s_prev
        else:
            # cooperated and dissatisfied -> decrease p (because p + p*s, s negative)
            p_new = p + p * s_prev
    else:  # action_prev == D
        if s_prev >= 0.0:
            # defected and satisfied -> decrease p (keep defecting)
            p_new = p - p * s_prev
        else:
            # defected and dissatisfied -> increase p (because - (1-p)*s, s negative => plus)
            p_new = p - (1.0 - p) * s_prev

    return fix01(p_new)


def bm_step(p: float, action_prev: int, reward_prev: float, A: float, beta: float):
    # computes stimulus from (reward_prev, A, beta) then applies BM update
    s_prev = stimulus(reward_prev, A=A, beta=beta)
    return bm_update(p=p, action_prev=action_prev, s_prev=s_prev)