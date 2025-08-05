import numpy as np

# --- Constants for Value Iteration ---
CONVERGENCE_TOLERANCE = 1e-4

def probability_weighting(p, gamma):
    """
    Applies a probability weighting function from Cumulative Prospect Theory (CPT).

    This function distorts objective probabilities into subjective decision weights.
    
    Args:
        p (float): The objective probability (0 to 1).
        gamma (float): The weighting parameter (typically between 0 and 1).

    Returns:
        float: The subjective decision weight.
    """
    return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1 / gamma))

def cpt_value_function(x, theta, omega, eta):
    """
    Applies the value function from Cumulative Prospect Theory (CPT).

    This function models how individuals perceive gains and losses, featuring
    diminishing sensitivity and loss aversion.

    Args:
        x (float): The objective value (gain or loss) relative to a reference point.
        theta (float): Risk aversion parameter for gains (typically < 1).
        omega (float): Loss aversion parameter (typically > 1).
        eta (float): Risk aversion parameter for losses (typically < 1).

    Returns:
        float: The subjective value (utility) of the outcome.
    """
    if x >= 0:
        return x**theta
    return -omega * (-x)**eta

def compute_new_wealth(w, wealth_update_scale, utility_val):
    """Calculates the agent's new wealth based on a utility-driven adjustment."""
    delta = utility_val - w
    return w + wealth_update_scale * delta

def compute_health_delta(h):
    """Calculates the potential change in health from an investment or decline."""
    k = np.log(10) / 150
    return (10 * np.exp(-k * h) + 1).astype(int)

def compute_health_cost(h):
    """Calculates the cost of investing to improve health."""
    return -compute_health_delta(h) + 11

def utility(w, h, alpha, rate=1.0):
    """Calculates the Cobb-Douglas utility from wealth and health."""
    return w**alpha * h**(rate - alpha)

def _calculate_invest_value(wealth, health, value_function, params):
    """Helper to calculate the expected value of the 'invest' action."""
    invest_cost = compute_health_cost(health)
    if wealth <= invest_cost:
        return -np.inf  # Cannot afford to invest

    health_delta = compute_health_delta(health)
    reference_utility = utility(wealth, health, params['alpha'])

    # State after paying cost
    wealth_after_cost = wealth - invest_cost
    
    # --- Successful Investment ---
    health_success = min(health + health_delta, params['max_state_value'])
    wealth_success = min(int(compute_new_wealth(wealth_after_cost, params['w_delta_scale'], utility(wealth_after_cost, health_success, params['alpha']))), params['max_state_value'])
    utility_success = utility(wealth_success, health_success, params['alpha'])
    cpt_delta_success = cpt_value_function(utility_success - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_success = value_function[wealth_success - 1, health_success - 1]

    # --- Failed Investment (health does not change) ---
    wealth_fail = min(int(compute_new_wealth(wealth_after_cost, params['w_delta_scale'], utility(wealth_after_cost, health, params['alpha']))), params['max_state_value'])
    utility_fail = utility(wealth_fail, health, params['alpha'])
    cpt_delta_fail = cpt_value_function(utility_fail - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_fail = value_function[wealth_fail - 1, health - 1]

    # Weighted sum of immediate CPT utility and future value
    immediate_cpt_value = params['cpt_P_increase'] * cpt_delta_success + params['cpt_P_increase_complement'] * cpt_delta_fail
    expected_future_value = params['cpt_P_increase'] * future_val_success + params['cpt_P_increase_complement'] * future_val_fail
    
    return immediate_cpt_value + params['beta'] * expected_future_value

def _calculate_save_value(wealth, health, value_function, params):
    """Helper to calculate the expected value of the 'save' (not invest) action."""
    health_delta = compute_health_delta(health)
    reference_utility = utility(wealth, health, params['alpha'])

    # --- Health Declines ---
    health_decline = max(1, health - health_delta)
    wealth_decline = min(int(compute_new_wealth(wealth, params['w_delta_scale'], utility(wealth, health_decline, params['alpha']))), params['max_state_value'])
    utility_decline = utility(wealth_decline, health_decline, params['alpha'])
    cpt_delta_decline = cpt_value_function(utility_decline - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_decline = value_function[wealth_decline - 1, health_decline - 1]

    # --- Health Stays Steady ---
    wealth_steady = min(int(compute_new_wealth(wealth, params['w_delta_scale'], utility(wealth, health, params['alpha']))), params['max_state_value'])
    utility_steady = utility(wealth_steady, health, params['alpha'])
    cpt_delta_steady = cpt_value_function(utility_steady - reference_utility, params['theta'], params['omega'], params['eta'])
    future_val_steady = value_function[wealth_steady - 1, health - 1]

    # Weighted sum
    immediate_cpt_value = params['cpt_P_decrease'] * cpt_delta_decline + params['cpt_P_decrease_complement'] * cpt_delta_steady
    expected_future_value = params['cpt_P_decrease'] * future_val_decline + params['cpt_P_decrease_complement'] * future_val_steady

    return immediate_cpt_value + params['beta'] * expected_future_value


def value_iteration(max_state_value, alpha, gamma, theta, omega, eta, beta, P_H_increase, w_delta_scale, P_H_decrease):
    """
    Computes the optimal health investment policy for an agent using value iteration.

    This algorithm solves for the optimal action (invest in health or not) for every
    possible state of (wealth, health), based on the principles of CPT.

    Args:
        max_state_value (int): The maximum value for wealth and health (e.g., 100).
        alpha (float): Preference for wealth vs. health in the utility function.
        gamma (float): Probability weighting parameter.
        theta (float): CPT risk aversion for gains.
        omega (float): CPT loss aversion.
        eta (float): CPT risk aversion for losses.
        beta (float): Future discount factor.
        P_H_increase (float): Probability of health increasing if investing.
        w_delta_scale (float): Scaling factor for wealth updates.
        P_H_decrease (float): Probability of health decreasing if not investing.

    Returns:
        np.ndarray: A (max_state_value x max_state_value) policy matrix where
                    policy[w, h] is 1 (invest) or 0 (save).
    """
    value_function = np.zeros((max_state_value, max_state_value))
    policy = np.zeros((max_state_value, max_state_value), dtype=int)
    
    # Pre-calculate CPT probabilities and bundle parameters
    params = {
        'alpha': alpha, 'theta': theta, 'omega': omega, 'eta': eta,
        'beta': beta, 'w_delta_scale': w_delta_scale, 'max_state_value': max_state_value,
        'cpt_P_increase': probability_weighting(P_H_increase, gamma),
        'cpt_P_increase_complement': probability_weighting(1 - P_H_increase, gamma),
        'cpt_P_decrease': probability_weighting(P_H_decrease, gamma),
        'cpt_P_decrease_complement': probability_weighting(1 - P_H_decrease, gamma)
    }

    norm = np.inf
    while norm > CONVERGENCE_TOLERANCE:
        old_value_function = value_function.copy()
        
        for w_idx in range(max_state_value):
            for h_idx in range(max_state_value):
                wealth, health = w_idx + 1, h_idx + 1  # States are 1-based

                # Calculate the value of each possible action
                invest_value = _calculate_invest_value(wealth, health, old_value_function, params)
                save_value = _calculate_save_value(wealth, health, old_value_function, params)

                # Choose the best action
                if invest_value > save_value:
                    value_function[w_idx, h_idx] = invest_value
                    policy[w_idx, h_idx] = 1
                else:
                    value_function[w_idx, h_idx] = save_value
                    policy[w_idx, h_idx] = 0
        
        norm = np.linalg.norm(value_function - old_value_function)

    return policy