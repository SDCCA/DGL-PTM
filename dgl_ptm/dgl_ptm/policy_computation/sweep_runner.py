# policy_computation/sweep_runner.py

import os
from tqdm import tqdm

# Note the new import paths
from simulation_analysis.experiment_config import (
    COST_SUBSIDY_FACTORS, EFFICACY_MULTIPLIERS, POLICY_DIR,
    get_policy_path, INFECTION_RISK_LEVELS
)
from dgl_ptm.config import SVEIRConfig
from .generator import create_and_save_policy_library

def compute_all_policies_for_sweep():
    """
    Orchestrates the batch pre-computation of all policy libraries needed for
    the full intervention sweep defined in experiment_config.py.
    """
    print("Starting batch pre-computation of all policy libraries...")
    print(f"Policy assets will be saved in: '{POLICY_DIR}'")

    base_config = SVEIRConfig()
    base_config.seed = 42

    os.makedirs(POLICY_DIR, exist_ok=True)
    
    total_runs = len(EFFICACY_MULTIPLIERS) * len(COST_SUBSIDY_FACTORS)
    pbar = tqdm(total=total_runs, desc="Policy Sets Generated")

    for efficacy in EFFICACY_MULTIPLIERS:
        for subsidy in COST_SUBSIDY_FACTORS:
            policy_path = get_policy_path(efficacy, subsidy)
            
            if os.path.exists(policy_path):
                print(f"Skipping already computed policy: {policy_path}")
                pbar.update(1)
                continue
            
            current_config = base_config.model_copy(deep=True)
            current_config.policy_library_path = policy_path
            current_config.steering_parameters.efficacy_multiplier = efficacy
            current_config.steering_parameters.cost_subsidy_factor = subsidy

            try:
                create_and_save_policy_library(
                    config=current_config,
                    infection_risk_levels=INFECTION_RISK_LEVELS
                )
            except Exception as e:
                print(f"Failed to generate policy for E={efficacy}, S={subsidy}: {e}", exc_info=True)
            
            pbar.update(1)
    
    pbar.close()
    print("All policy libraries have been computed.")
