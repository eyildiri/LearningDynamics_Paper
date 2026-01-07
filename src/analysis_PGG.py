import numpy as np

# Fig 2A: mean ± SD over trials

def coop_timecourse_mean_sd(actions_all):
    """
    actions_all: (n_trials, tmax, n_agents) with 0/1 actions
    Returns:
      mean_t: (tmax,) mean fraction coop at each t
      sd_t:   (tmax,) SD across trials (paper-style shaded area)
    """
    frac_trial_t = actions_all.mean(axis=2)  # (n_trials, tmax)
    mean_t = frac_trial_t.mean(axis=0)
    sd_t = frac_trial_t.std(axis=0, ddof=1) if actions_all.shape[0] > 1 else np.zeros_like(mean_t)
    return mean_t, sd_t

# Fig 2B heatmap
def mean_coop(actions_all, t_start=0, t_end=25):
    """Mean cooperation over trials, time window, agents."""
    return actions_all[:, t_start:t_end, :].mean()


# Helper: neighbors and fC(t-1)
def neighbors_to_array(neighbors):
    """neighbors: list of lists length n_agents, each list length 4."""
    neigh = np.asarray(neighbors, dtype=int)
    if neigh.ndim != 2 or neigh.shape[1] != 4:
        raise ValueError("Expected neighbors shape (n_agents, 4).")
    return neigh
def fC_prev(actions_prev, neigh_arr):
    """
    actions_prev: (n_agents,) 0/1
    neigh_arr: (n_agents, 4)
    Returns fC: (n_agents,) in {0,0.25,0.5,0.75,1}
    """
    return actions_prev[neigh_arr].mean(axis=1)

# Fig 3: CC / MCC 

def cc_mcc(actions_all, t_start=1, t_end=25, tresh=0.5):
    """
    Compute CC/MCC curves aggregated across all trials/time/agents.
    Returns:
      f_vals: [0,0.25,0.5,0.75,1]
      pC:     P(C_t | fC_{t-1}) : !!!conditional probability of cooperating at t given fraction of cooperating neighbors at t-1!!!
      pC_C:   P(C_t | prev=C, fC_{t-1}) 
      pC_D:   P(C_t | prev=D, fC_{t-1})
      counts: dict of counts (global/prevC/prevD) per f bin
    """
    n_trials, tmax, n_agents = actions_all.shape

    # neighbors in array (n_agents, 4)


    neigh = np.linspace(0, n_agents-1, n_agents, dtype=int).reshape(-1,1)

    f_vals = np.linspace(0, 1, 14)[1:-1]

    sum_global = np.zeros(12, dtype=float)
    cnt_global = np.zeros(12, dtype=float)

    sum_prevC = np.zeros(12, dtype=float)
    cnt_prevC = np.zeros(12, dtype=float)

    sum_prevD = np.zeros(12, dtype=float)
    cnt_prevD = np.zeros(12, dtype=float)

    for tr in range(n_trials):
        for t in range(t_start, t_end):
            prev_a = actions_all[tr, t-1, :]  
            now_a  = actions_all[tr, t,   :]  
        

            total_prev = prev_a.sum()
            n = len(prev_a)
            average_other = (total_prev - prev_a) / (n - 1)
       

            #for played_bet in now_a:
            for i in range(len(now_a)):

                played_bet = now_a[i]

                idx = np.argmin(np.abs(f_vals - average_other[i]))
              
                cnt_global[idx] += 1
                sum_global[idx] += played_bet
                
                if prev_a[i] >= tresh:
                    cnt_prevC[idx] += 1
                    sum_prevC[idx] += played_bet
                
                else:
                    cnt_prevD[idx] += 1
                    sum_prevD[idx] += played_bet


    def safe_div(num, den):
        # exp : pC[k] = sum_global[k] / cnt_global[k]
        out = np.full_like(num, np.nan, dtype=float)
        m = den > 0
        out[m] = num[m] / den[m]
        return out

    return {
        "f_vals": f_vals,
        "pC":   safe_div(sum_global, cnt_global),
        "pC_C": safe_div(sum_prevC,  cnt_prevC),
        "pC_D": safe_div(sum_prevD,  cnt_prevD),
        "counts": {"global": cnt_global, "prevC": cnt_prevC, "prevD": cnt_prevD},
    }


# Fig 4: alpha1/alpha2

def alpha_fit_polyfit(actions_all, condition=None, t_start=1, t_end=25, tresh=0.5):
    """
    Collect every (x,y) point satisfying the condition then fit y ~ a1*x + a2. with np.polyfit.

    x = fC(t-1) = fraction of cooperating neighbors at previous turn in {0,0.25,0.5,0.75,1}
    y = action_t (0/1)

    Returns:
      a1, a2, n_samples
    """

    n_trials, tmax, n_agents = actions_all.shape
    neigh = np.linspace(0, n_agents-1, n_agents, dtype=int).reshape(-1,1)

    xs = []
    ys = []

    for tr in range(n_trials):
        for t in range(t_start, t_end):

            

            prev_a = actions_all[tr, t-1, :]  
            now_a  = actions_all[tr, t,   :]  

            total_prev = prev_a.sum()
            n = len(prev_a)
            average_other = (total_prev - prev_a) / (n - 1)



            #for played_bet in now_a:
            for i in range(len(now_a)):
                if condition == "prevC" and prev_a[i] - tresh >=0:
                    xs.append(np.array([average_other[i]]))
                    ys.append(np.array([now_a[i]]))
                elif condition == "prevD" and prev_a[i] - tresh <0:
                    xs.append(np.array([average_other[i]]))
                    ys.append(np.array([now_a[i]]))
                elif condition is None:
                    xs.append(np.array([average_other[i]]))
                    ys.append(np.array([now_a[i]]))



    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    # Fit y ~ a1*x + a2
    a1, a2 = np.polyfit(x_all, y_all, 1)
    return float(a1), float(a2), int(x_all.size)
