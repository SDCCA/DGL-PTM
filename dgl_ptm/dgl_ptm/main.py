# main.py

import argparse

# Import the main functions from our new, clean modules
from policy_computation.sweep_runner import compute_all_policies_for_sweep
from simulation_analysis.intervention_sweep import run_sweep_and_generate_heatmap

def main():
    """
    Provides a command-line interface to run the different stages of the
    SVEIR model experimental workflow.
    """
    parser = argparse.ArgumentParser(description="Run stages of the SVEIR model experiment.")
    parser.add_argument(
        'stage',
        choices=['precompute', 'simulate'],
        help="The stage of the experiment to run: 'precompute' to generate all policy files, 'simulate' to run simulations and create the heatmap."
    )
    
    args = parser.parse_args()

    if args.stage == 'precompute':
        print("--- STAGE 1: Starting Policy Pre-computation ---")
        compute_all_policies_for_sweep()
        print("--- STAGE 1: Policy Pre-computation Finished ---")
    
    elif args.stage == 'simulate':
        print("--- STAGE 2: Starting Simulation and Analysis ---")
        run_sweep_and_generate_heatmap()
        print("--- STAGE 2: Simulation and Analysis Finished ---")

if __name__ == "__main__":
    main()