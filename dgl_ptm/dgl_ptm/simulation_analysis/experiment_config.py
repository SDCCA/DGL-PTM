# simulation_analysis/experiment_config.py

import numpy as np
import os

# --- Define the Parameter Space for the Interventions ---
# This grid defines the axes of your final heatmap.
COST_SUBSIDY_FACTORS = np.linspace(1.0, 0.4, 5)  # From no subsidy to a 60% subsidy
EFFICACY_MULTIPLIERS = np.linspace(1.0, 2.0, 5)  # From normal to double efficacy

# --- Define the Policy Space Parameters ---
# Defines the granularity of the policy space computed for the analysis.
INFECTION_RISK_LEVELS = np.array([0.01])

# --- Define File and Directory Settings ---
# The central directory where all generated policy libraries will be stored.
POLICY_DIR = "policy_libraries_sweep"

# --- Helper Function for Consistent Naming ---
def get_policy_path(efficacy: float, subsidy: float) -> str:
    """
    Generates a standardized, unique file path for a policy library
    based on its intervention parameters.
    """
    filename = f"policy_eff_{efficacy:.2f}_cost_{subsidy:.2f}.npz"
    return os.path.join(POLICY_DIR, filename)