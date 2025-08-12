import numpy as np
import os

# --- Define the Parameter Space for the Interventions ---
# This grid defines the axes of your final heatmap.
COST_SUBSIDY_FACTORS = np.linspace(1.0, 0.4, 5)  # From no subsidy to a 60% subsidy
EFFICACY_MULTIPLIERS = np.linspace(1.0, 2.0, 5)  # From normal to double efficacy

INFECTION_RISK_LEVELS = np.array([0.01, 0.03, 0.05, 0.08, 0.12])

# --- Define File and Directory Settings ---
POLICY_DIR = "policy_libraries_sweep"
RESULTS_DIR = "simulation_results"

# Add a dedicated directory for individual run outputs
SIM_RUNS_DIR = "simulation_runs"

# --- Helper Functions for Consistent Naming ---
def get_policy_path(efficacy: float, subsidy: float) -> str:
    """Generates a standardized path for a policy library."""
    filename = f"policy_eff_{efficacy:.2f}_cost_{subsidy:.2f}.npz"
    return os.path.join(POLICY_DIR, filename)

def get_results_path(number_agents: int, repetitions: int) -> str:
    """Generates a standardized path for a simulation summary results grid."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"summary_grid_prop_infected_agents_{number_agents}reps{repetitions}.npy"
    return os.path.join(RESULTS_DIR, filename)

def get_full_results_path(number_agents: int, repetitions: int) -> str:
    """Generates a standardized path for the full, detailed simulation results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"full_results_agents_{number_agents}_reps_{repetitions}.pkl"
    return os.path.join(RESULTS_DIR, filename)