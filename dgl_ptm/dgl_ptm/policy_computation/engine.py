# policy_computation/engine.py

import numpy as np

# --- Constants for Value Iteration ---
CONVERGENCE_TOLERANCE = 1e-4

def probability_weighting(p, gamma):
    """
    Applies a probability weighting function from Cumulative Prospect Theory (CPT).
    """
    return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1 / gamma))

def cpt_value_function(x, theta, omega, eta):
    """
    Applies the value function from Cumulative Prospect Theory (CPT).
    """
    if x >= 0:
        return x**theta
    return -omega * (-x)**eta

def compute_new_wealth(w, wealth_update_scale, utility_val):
    """Calculates the agent's new wealth based on a utility-driven adjustment."""
    delta = utility_val - w
    return w + wealth_update_scale * delta

def utility(w, h, alpha, rate=1.0):
    """Calculates the Cobb-Douglas utility from wealth and health."""
    # Add a small epsilon to prevent log(0) or power of zero issues if w or h is 0
    return (w + 1e-6)**alpha * (h + 1e-6)**(rate - alpha)

# --- Functions for Health/Cost Dynamics ---

def compute_health_delta(h, params):
    """Calculates the POSITIVE change in health from a successful investment."""
    k = np.log(10) / 150
    base_delta = 10 * np.exp(-k * h) + 1
    return base_delta * params.get('efficacy_multiplier', 1.0)

def compute_health_decline(h):
    """Calculates the potential health decline from not investing (natural decay)."""
    k = np.log(10) / 150
    return 10 * np.exp(-k * h) + 1

def compute_health_cost(h, params):
    """Calculates the cost of investing to improve health."""
    base_cost = -compute_health_delta(h, params) + 11
    return base_cost * params.get('cost_subsidy_factor', 1.0)


def _calculate_expected_value(wealth, health, value_function, params, action_health_change, action_health_prob, action_cost):
    """
    Generic helper to calculate the expected value of an action, including external risk.
    """
    if wealth <= action_cost:
        return -np.inf

    wealth_after_cost = wealth - action_cost
    reference_utility = utility(wealth, health, params['alpha'])

    # Outcome 1: Health changes as expected
    health_1 = min(max(1, health + action_health_change), params['max_state_value'])
    wealth_1 = min(int(compute_new_wealth(wealth_after_cost, params['wealth_update_A'], utility(wealth_after_cost, health_1, params['alpha']))), params['max_state_value'])
    cpt_delta_1 = cpt_value_function(utility(wealth_1, health_1, params['alpha']) - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_1 = value_function[wealth_1 - 1, health_1 - 1]

    # Outcome 2: Health remains steady
    health_2 = health
    wealth_2 = min(int(compute_new_wealth(wealth_after_cost, params['wealth_update_A'], utility(wealth_after_cost, health_2, params['alpha']))), params['max_state_value'])
    cpt_delta_2 = cpt_value_function(utility(wealth_2, health_2, params['alpha']) - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_2 = value_function[wealth_2 - 1, health_2 - 1]
    
    cpt_prob_1 = probability_weighting(action_health_prob, params['gamma'])
    cpt_prob_2 = probability_weighting(1 - action_health_prob, params['gamma'])
    
    immediate_cpt_value = cpt_prob_1 * cpt_delta_1 + cpt_prob_2 * cpt_delta_2
    expected_future_value = cpt_prob_1 * future_val_1 + cpt_prob_2 * future_val_2
    
    health_susceptibility = np.exp(-params["infection_reduction_factor_per_health_unit"] * (health - 1.0))
    prob_infection = np.clip(params['global_infection_prob'] * health_susceptibility, 0, 1)

    health_if_infected = max(1, health - 20)
    wealth_if_infected = max(1, wealth - 20)
    
    value_if_infected = utility(wealth_if_infected, health_if_infected, params['alpha']) + params['beta'] * value_function[wealth_if_infected - 1, health_if_infected - 1]
    
    final_value = (1 - prob_infection) * (immediate_cpt_value + params['beta'] * expected_future_value) + \
                  (prob_infection) * cpt_value_function(value_if_infected - reference_utility, params['theta'], params['omega'], params['eta'])

    return final_value


def value_iteration(max_state_value, alpha, gamma, theta, omega, eta, beta, params):
    """
    Computes the optimal health investment policy for an agent using value iteration.
    """
    value_function = np.zeros((max_state_value, max_state_value))
    policy = np.zeros((max_state_value, max_state_value), dtype=int)
    
    local_params = params.copy()
    local_params.update({'alpha': alpha, 'gamma': gamma, 'beta': beta, 'max_state_value': max_state_value,
                         'theta': theta, 'omega': omega, 'eta': eta})

    P_H_increase = params['P_H_increase']
    P_H_decrease = params['P_H_decrease']

    norm = np.inf
    while norm > CONVERGENCE_TOLERANCE:
        old_value_function = value_function.copy()
        
        health_delta_array = compute_health_delta(np.arange(1, max_state_value + 1), local_params).astype(int)
        health_decline_array = compute_health_decline(np.arange(1, max_state_value + 1)).astype(int)
        invest_cost_array = compute_health_cost(np.arange(1, max_state_value + 1), local_params)

        for w_idx in range(max_state_value):
            for h_idx in range(max_state_value):
                wealth, health = w_idx + 1, h_idx + 1

                invest_value = _calculate_expected_value(
                    wealth, health, old_value_function, local_params,
                    action_health_change=health_delta_array[h_idx],
                    action_health_prob=P_H_increase,
                    action_cost=invest_cost_array[h_idx]
                )
                
                save_value = _calculate_expected_value(
                    wealth, health, old_value_function, local_params,
                    action_health_change=-health_decline_array[h_idx],
                    action_health_prob=P_H_decrease,
                    action_cost=0
                )

                if invest_value > save_value:
                    value_function[w_idx, h_idx] = invest_value
                    policy[w_idx, h_idx] = 1
                else:
                    value_function[w_idx, h_idx] = save_value
                    policy[w_idx, h_idx] = 0
        
        norm = np.linalg.norm(value_function - old_value_function)

    return policy