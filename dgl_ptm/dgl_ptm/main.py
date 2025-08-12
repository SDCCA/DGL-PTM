import argparse
import os

from policy_computation.sweep_runner import compute_all_policies_for_sweep
from simulation_analysis.intervention_sweep import run_simulation_sweep, generate_heatmap
from simulation_analysis.experiment_config import get_results_path
from dgl_ptm.environment.grid_generator import create_and_save_realistic_grid
from dgl_ptm.config import SVEIRCONFIG

def main():
    """
    Provides a command-line interface to run the different stages of the
    SVEIR model experimental workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run stages of the SVEIR model experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'stage',
        choices=['precompute', 'simulate', 'plot-heatmap', 'plot-curves', 'create-grid'],
        help=(
            "The stage of the experiment to run:\n"
            "  'create-grid'  - Generate the realistic base grid from real-world data (run once).\n"
            "  'precompute'   - Generate all policy files.\n"
            "  'simulate'     - Run simulations and save the results.\n"
            "  'plot-heatmap' - Generate the summary heatmap from saved results.\n"
        )
    )

    # Arguments for 'simulate' and 'plot' stages
    parser.add_argument(
        '-n', '--agents',
        type=int,
        default=250,
        help="Number of agents to use in the simulation (default: 250)."
    )
    parser.add_argument(
        '-r', '--repetitions',
        type=int,
        default=5,
        help="Number of repetitions for each intervention scenario (default: 5)."
    )
    parser.add_argument(
        '-c', '--cores',
        type=int,
        default=6,
        help="Number of CPU cores to use for parallel processing (default: 6)."
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=SVEIRCONFIG.step_target, # Default to the value in the config
        help=f"Number of steps to run each simulation (default: {SVEIRCONFIG.step_target})."
    )
    
    args = parser.parse_args()

    if args.stage == 'create-grid':
        create_and_save_realistic_grid()

    elif args.stage == 'precompute':
        print("--- Stage: Pre-computing Policies ---")
        compute_all_policies_for_sweep()

    elif args.stage == 'simulate':
        print(f"--- Stage: Running Simulation Sweep ---")
        print(f"Parameters: {args.agents} agents, {args.steps} steps, {args.repetitions} repetition(s) per scenario, using {args.cores} cores.")
        run_simulation_sweep(
            number_agents=args.agents,
            repetitions=args.repetitions,
            num_cores=args.cores,
            steps=args.steps
        )

    elif args.stage == 'plot-heatmap':
        print("--- Stage: Generating Heatmap ---")
        results_grid_path = get_results_path(args.agents, args.repetitions)

        if not os.path.exists(results_grid_path):
            print(f"Error: Results grid file not found at '{results_grid_path}'")
            print("Please run the 'simulate' stage first.")
            return

        generate_heatmap(results_grid_path)

if __name__ == "__main__":
    main()