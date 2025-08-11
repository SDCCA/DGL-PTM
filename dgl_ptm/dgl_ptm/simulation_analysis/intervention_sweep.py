# simulation_analysis/intervention_sweep.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from .experiment_config import (
    COST_SUBSIDY_FACTORS, EFFICACY_MULTIPLIERS, get_policy_path, POLICY_DIR
)
from dgl_ptm.config import SVEIRConfig
from dgl_ptm.model.initialize_model import SVEIRModel

def run_sweep_and_generate_heatmap():
    """
    Runs the full simulation sweep using pre-computed policies and generates
    the final analysis heatmap.
    """
    print("Starting simulation sweep using pre-computed policies...")
    
    results_grid = np.zeros((len(EFFICACY_MULTIPLIERS), len(COST_SUBSIDY_FACTORS)))
    
    base_config = SVEIRConfig()
    base_config.num_agent_personas = 16
    base_config.step_target = 150
    base_config.number_agents = 250
    base_config.seed = 42

    for i, efficacy in enumerate(tqdm(EFFICACY_MULTIPLIERS, desc="Efficacy Levels")):
        for j, subsidy in enumerate(COST_SUBSIDY_FACTORS):
            run_name = f"sim_eff_{efficacy:.2f}_cost_{subsidy:.2f}"
            print(f"--- Starting: {run_name} ---")

            policy_path = get_policy_path(efficacy, subsidy)

            if not os.path.exists(policy_path):
                print(f"Policy file not found for {run_name}! Path: {policy_path}")
                print("Please run the 'precompute' stage first. Skipping this run.")
                results_grid[i, j] = -1
                continue
            
            current_config = base_config.model_copy(deep=True)
            current_config.policy_library_path = policy_path
            current_config.steering_parameters.efficacy_multiplier = efficacy
            current_config.steering_parameters.cost_subsidy_factor = subsidy
            
            try:
                model = SVEIRModel(model_identifier=run_name)
                model.set_model_parameters(**current_config.model_dump())
                model.initialize_model(verbose=False)
                model.run()
                
                total_infections = model.get_total_infections()
                results_grid[i, j] = total_infections
                print(f"Finished Run. Total Infections: {total_infections}")

            except Exception as e:
                print(f"ERROR during simulation for {run_name}: {e}", exc_info=True)
                results_grid[i, j] = -1

    print("All simulations complete. Generating heatmap...")
    results_grid_flipped = np.flipud(results_grid)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        results_grid_flipped, annot=True, fmt=".0f", cmap="plasma_r",
        xticklabels=[f"{x:.2f}" for x in COST_SUBSIDY_FACTORS],
        yticklabels=[f"{y:.2f}" for y in reversed(EFFICACY_MULTIPLIERS)]
    )
    ax.set_title("Impact of Interventions on Total Infections over 150 Steps", fontsize=16, pad=20)
    ax.set_xlabel("Cost Subsidy Factor (Lower is Cheaper Healthcare)", fontsize=12)
    ax.set_ylabel("Health Efficacy Multiplier (Higher is Better Healthcare)", fontsize=12)
    
    output_filename = "final_intervention_heatmap.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Heatmap saved to {output_filename}")
    plt.show()