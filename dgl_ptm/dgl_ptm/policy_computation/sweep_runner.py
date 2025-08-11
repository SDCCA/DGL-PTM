# policy_computation/sweep_runner.py

import os
from tqdm import tqdm
import multiprocessing
from functools import partial

# Import our shared experiment settings and the necessary model components
from simulation_analysis.experiment_config import (
    COST_SUBSIDY_FACTORS,
    EFFICACY_MULTIPLIERS,
    POLICY_DIR,
    get_policy_path,
    INFECTION_RISK_LEVELS
)
from dgl_ptm.config import SVEIRConfig
from .generator import create_and_save_policy_library

# --- 1. The Worker Function ---
# We define the work for a SINGLE job in this function.
# It takes the base config and the specific parameters for one run.
def process_one_policy_set(task_params: tuple, base_config: SVEIRConfig):
    """
    This is the "worker" function that will be executed by each parallel process.
    It handles the complete logic for generating one policy file.

    Args:
        task_params (tuple): A tuple containing the (efficacy, subsidy) for this job.
        base_config (SVEIRConfig): The base model configuration template.
    """
    efficacy, subsidy = task_params
    
    # Determine the standardized path for this policy file
    policy_path = get_policy_path(efficacy, subsidy)
    
    # Check if the file already exists (for resumability)
    if os.path.exists(policy_path):
        # We return a status so the main process knows what happened.
        return f"Skipped: {policy_path}"
    
    current_config = base_config.model_copy(deep=True)
    current_config.policy_library_path = policy_path
    current_config.steering_parameters.efficacy_multiplier = efficacy
    current_config.steering_parameters.cost_subsidy_factor = subsidy

    try:
        create_and_save_policy_library(
            config=current_config,
            infection_risk_levels=INFECTION_RISK_LEVELS
        )
        return f"Success: {policy_path}"
    except Exception as e:
        # Log the full error with traceback for debugging
        print(f"Failed to generate policy for E={efficacy}, S={subsidy}: {e}", exc_info=True)
        return f"Failed: {policy_path}"

# --- 2. The Main Orchestrator Function ---
# This function now sets up the pool and distributes the work.
def compute_all_policies_for_sweep():
    """
    Orchestrates the PARALLEL batch pre-computation of all policy libraries
    needed for the full intervention sweep.
    """
    print("Starting PARALLEL batch pre-computation of all policy libraries...")
    
    # --- Setup ---
    # You can change the number of cores here
    NUM_CORES = 6
    print(f"Using up to {NUM_CORES} CPU cores.")

    base_config = SVEIRConfig()
    base_config.seed = 42

    os.makedirs(POLICY_DIR, exist_ok=True)
    
    # --- Create the full list of tasks ---
    # Each task is a tuple of the parameters that define a single job.
    tasks = [(efficacy, subsidy) for efficacy in EFFICACY_MULTIPLIERS for subsidy in COST_SUBSIDY_FACTORS]
    
    # --- Filter out tasks that are already complete ---
    # This is more efficient than letting each worker check individually.
    incomplete_tasks = [
        task for task in tasks if not os.path.exists(get_policy_path(task[0], task[1]))
    ]
    
    if not incomplete_tasks:
        print("All policy libraries have already been computed. Nothing to do.")
        return
        
    print(f"Found {len(incomplete_tasks)} policy sets to generate out of {len(tasks)} total.")

    # --- Distribute tasks to the process pool ---
    # We use `functools.partial` to "bake" the base_config into our worker function.
    # This is a clean way to pass constant arguments to a function used in a map.
    worker = partial(process_one_policy_set, base_config=base_config)

    # The 'with' statement ensures the pool is properly shut down even if errors occur.
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        # `pool.imap` is a lazy version of `map`. It's good for memory and lets tqdm update progress
        # as each job finishes, not when all jobs are submitted.
        results = list(tqdm(pool.imap(worker, incomplete_tasks), total=len(incomplete_tasks), desc="Policy Sets Generated"))

    print("--- Batch processing complete. Summary ---")
    # Optional: Print a summary of what happened
    success_count = sum(1 for r in results if r.startswith("Success"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))
    print(f"Successfully generated: {success_count} policy sets.")
    if failed_count > 0:
        print(f"Failed to generate: {failed_count} policy sets. Check logs for details.")
    
    print("All policy libraries have been computed.")