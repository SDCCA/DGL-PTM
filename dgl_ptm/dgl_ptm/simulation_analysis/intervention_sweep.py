# simulation_analysis/intervention_sweep.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing
from tqdm import tqdm
import time
import pickle
import traceback

from .experiment_config import (
    COST_SUBSIDY_FACTORS, EFFICACY_MULTIPLIERS, get_policy_path,
    get_results_path, get_full_results_path, get_sim_runs_path
)
from dgl_ptm.config import SVEIRConfig
from dgl_ptm.model.initialize_model import SVEIRModel

def run_single_simulation(params: dict) -> dict:
    """
    Worker function: runs a single simulation instance and returns the result.
    """
    run_name = params['run_name']
    sim_runs_path = params['sim_runs_path']
    config_dict = params["config_dict"]

    try:
        model = SVEIRModel(model_identifier=run_name, root_path=sim_runs_path)
        config_for_model = config_dict.copy()
        config_for_model = config_dict.copy()
        config_for_model.pop('model_identifier', None)
        model.set_model_parameters(**config_for_model)
        model.initialize_model(verbose=False)
        model.run()
        
        time_series_data = model.get_time_series_data()
        final_states = model.get_final_agent_states()
        incidence_curve = time_series_data['incidence']
        prevalence_curve = time_series_data['prevalence']

        return {
            'efficacy': config_dict['steering_parameters']['efficacy_multiplier'],
            'subsidy': config_dict['steering_parameters']['cost_subsidy_factor'],
            'total_infections': model.get_total_infections(),
            'peak_incidence': max(incidence_curve) if incidence_curve else 0,
            'incidence_curve': incidence_curve,
            'prevalence_curve': prevalence_curve,
            'proportion_infected': model.get_proportion_infected_at_least_once(),
            'final_health': final_states['health'],
            'final_wealth': final_states['wealth']
        }

    except Exception:
        print(f"\n--- ERROR IN WORKER: {run_name} ---")
        traceback.print_exc()
        print(f"--- END ERROR ---")
        return {
            'efficacy': config_dict.get('steering_parameters', {}).get('efficacy_multiplier', -1),
            'subsidy': config_dict.get('steering_parameters', {}).get('cost_subsidy_factor', -1),
            'total_infections': -1,
            'peak_incidence': -1,
            'incidence_curve': [],
            'prevalence_curve': [],
            'proportion_infected': -1.0,
            'final_health': [],
            'final_wealth': []
        }


def worker_unpacker(args):
    """Helper to unpack arguments for the pool."""
    return run_single_simulation(args)


def run_simulation_sweep(number_agents: int, repetitions: int, num_cores: int, steps: int, experiment_name: str, grid_id: str, policy_set_id: str):
    """
    Runs the full simulation sweep in parallel and saves both detailed and summary results.
    """
    print(f"Starting parallel simulation sweep for experiment: {experiment_name}")
    sim_runs_path = get_sim_runs_path(experiment_name)
    print(f"Individual run outputs will be saved in: {sim_runs_path}")

    tasks = []
    base_seed = SVEIRConfig().seed

    for i, efficacy in enumerate(EFFICACY_MULTIPLIERS):
        for j, subsidy in enumerate(COST_SUBSIDY_FACTORS):
            policy_path = get_policy_path(policy_set_id, efficacy, subsidy)
            if not os.path.exists(policy_path):
                print(f"WARNING: Policy file not found at '{policy_path}'. Skipping combination.")
                continue

            for r in range(repetitions):
                unique_seed = base_seed + (i * len(COST_SUBSIDY_FACTORS) * repetitions) + (j * repetitions) + r
                run_name = f"sim_eff_{efficacy:.2f}_cost_{subsidy:.2f}_rep_{r+1}"

                # Create the complete configuration dictionary for this specific run
                config_for_run = {
                    "model_identifier": run_name,
                    "number_agents": number_agents,
                    "step_target": steps,
                    "seed": unique_seed,
                    "policy_library_path": policy_path,
                    "spatial_creation_args": {"grid_id": grid_id},
                    "steering_parameters": {
                        "efficacy_multiplier": efficacy,
                        "cost_subsidy_factor": subsidy
                    }
                }

                tasks.append({
                    'run_name': run_name,
                    'config_dict': config_for_run,
                    'sim_runs_path': sim_runs_path
                })
    
    if not tasks:
        print("No valid policy files found. Cannot run simulations.")
        return

    print(f"Total simulation runs to perform: {len(tasks)}\n")
    time.sleep(1)

    raw_results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(tasks), desc="Running Simulations") as pbar:
            for result in pool.imap_unordered(worker_unpacker, tasks):
                if result:
                    raw_results.append(result)
                pbar.update(1)
    
    full_results_path = get_full_results_path(experiment_name, number_agents, repetitions)
    with open(full_results_path, 'wb') as f:
        pickle.dump(raw_results, f)
    print(f"\nResults saved to: {full_results_path}")

    results_agg = {}
    for res in raw_results:
        key = (res['efficacy'], res['subsidy'])
        if key not in results_agg:
            results_agg[key] = []
        results_agg[key].append(res['proportion_infected'])

    summary_grid = np.zeros((len(EFFICACY_MULTIPLIERS), len(COST_SUBSIDY_FACTORS)))
    for i, efficacy in enumerate(EFFICACY_MULTIPLIERS):
        for j, subsidy in enumerate(COST_SUBSIDY_FACTORS):
            key = (efficacy, subsidy)
            proportions_list = results_agg.get(key, [-1.0])
            valid_proportions = [p for p in proportions_list if p >= 0]
            summary_grid[i, j] = np.mean(valid_proportions) if valid_proportions else -1
            
    summary_grid_path = get_results_path(experiment_name, number_agents, repetitions)
    np.save(summary_grid_path, summary_grid)


def generate_heatmap(results_grid_file: str):
    """
    Loads a saved summary grid and generates the analysis heatmap.
    """
    print(f"Loading results from {results_grid_file} to generate heatmap...")
    results_grid = np.load(results_grid_file)

    results_grid_flipped = np.flipud(results_grid)
    plt.figure(figsize=(12, 10))
    
    annot_data = np.char.mod('%.2f', results_grid_flipped)
    annot_data[results_grid_flipped < 0] = 'FAIL'

    ax = sns.heatmap(
        results_grid_flipped, annot=annot_data, fmt="s", cmap="plasma_r",
        xticklabels=[f"{x:.2f}" for x in COST_SUBSIDY_FACTORS],
        yticklabels=[f"{y:.2f}" for y in reversed(EFFICACY_MULTIPLIERS)],
        linewidths=.5, annot_kws={"size": 12}
    )
    ax.set_title("Impact of Interventions on Total Proportion of Population Infected (Attack Rate)", fontsize=16, pad=20)
    ax.set_xlabel("Cost Subsidy Factor (Lower is Cheaper Healthcare)", fontsize=12)
    ax.set_ylabel("Health Efficacy Multiplier (Higher is Better Healthcare)", fontsize=12)
    
    output_dir = os.path.dirname(results_grid_file)
    base_name = os.path.basename(results_grid_file)
    output_filename = os.path.join(output_dir, f"heatmap_{os.path.splitext(base_name)[0]}.png")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()


def plot_epidemic_curves(experiment_name: str, agents: int, repetitions: int):
    """
    Loads full simulation results and plots the mean number of currently
    infected agents over time (prevalence) for key scenarios.
    """
    full_results_file = get_full_results_path(experiment_name, agents, repetitions)
    if not os.path.exists(full_results_file):
        print(f"Error: Full results file not found at '{full_results_file}'.")
        print("Please ensure the 'simulate' stage was run for this experiment.")
        return
        
    print(f"Loading full results from {full_results_file}...")
    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    # --- Aggregate prevalence curves for each scenario ---
    curves_agg = {}
    for res in raw_results:
        # Ensure the prevalence_curve key exists from the new run
        if 'prevalence_curve' not in res:
            print("Error: 'prevalence_curve' not found in results. Please re-run the 'simulate' stage.")
            return

        key = (res['efficacy'], res['subsidy'])
        if key not in curves_agg:
            curves_agg[key] = []
        curves_agg[key].append(res['prevalence_curve'])

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define the key scenarios you want to compare
    scenarios_to_plot = {
        "Baseline (No Intervention)": (1.0, 1.0),
        "Best Intervention": (max(EFFICACY_MULTIPLIERS), min(COST_SUBSIDY_FACTORS)),
    }

    for label, (eff, sub) in scenarios_to_plot.items():
        # Find the key in your results that is numerically closest to the target scenario
        closest_key = min(curves_agg.keys(), key=lambda k: abs(k[0]-eff) + abs(k[1]-sub))
        
        curves = curves_agg.get(closest_key)
        if not curves:
            print(f"Warning: No data found for scenario '{label}'")
            continue
        
        # Curves can have slightly different lengths; pad them to the max length for averaging
        max_len = max(len(c) for c in curves)
        padded_curves = [np.pad(c, (0, max_len - len(c)), 'constant') for c in curves]
        
        mean_curve = np.mean(padded_curves, axis=0)
        std_curve = np.std(padded_curves, axis=0)
        timesteps = np.arange(len(mean_curve))
        
        # Plot the mean curve
        line, = ax.plot(timesteps, mean_curve, label=f"{label}\n(Eff: {closest_key[0]:.2f}, Sub: {closest_key[1]:.2f})", lw=2)
        # Add a shaded confidence interval (standard deviation)
        ax.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=line.get_color())

    ax.set_title("Impact of Interventions on Epidemic Progression (Prevalence)", fontsize=16, pad=15)
    ax.set_xlabel("Time (Days)", fontsize=12)
    ax.set_ylabel("Number of Currently Infected Agents", fontsize=12)
    ax.legend(title="Intervention Scenario", fontsize=10, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)

    # Save the figure
    output_dir = os.path.dirname(full_results_file)
    output_filename = os.path.join(output_dir, f"prevalence_curves_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)

    plt.show()


def plot_final_state_violins(experiment_name: str, agents: int, repetitions: int):
    """
    Loads full simulation results and creates violin plots comparing the
    distribution of final agent health and wealth for key scenarios.
    """
    full_results_file = get_full_results_path(experiment_name, agents, repetitions)
    if not os.path.exists(full_results_file):
        print(f"Error: Full results file not found at '{full_results_file}'.")
        return
        
    print(f"Loading full results from {full_results_file}...")
    with open(full_results_file, 'rb') as f:
        raw_results = pickle.load(f)

    # --- Define Scenarios and Aggregate Data ---
    baseline_key = min(raw_results, key=lambda r: abs(r['efficacy']-1.0) + abs(r['subsidy']-1.0))
    best_eff = max(r['efficacy'] for r in raw_results)
    best_sub = min(r['subsidy'] for r in raw_results)
    best_case_key = min(raw_results, key=lambda r: abs(r['efficacy']-best_eff) + abs(r['subsidy']-best_sub))

    scenarios = {
        "Baseline": (baseline_key['efficacy'], baseline_key['subsidy']),
        "Best Intervention": (best_case_key['efficacy'], best_case_key['subsidy'])
    }
    
    # --- Prepare data for Pandas DataFrame ---
    plot_data = []
    for res in raw_results:
        for scenario_name, (eff, sub) in scenarios.items():
            if res['efficacy'] == eff and res['subsidy'] == sub:
                # Add every agent's final health to the list
                for health_val in res.get('final_health', []):
                    plot_data.append({'value': health_val, 'Metric': 'Health', 'Scenario': scenario_name})
                # Add every agent's final wealth to the list
                for wealth_val in res.get('final_wealth', []):
                    plot_data.append({'value': wealth_val, 'Metric': 'Wealth', 'Scenario': scenario_name})

    if not plot_data:
        print("Error: Could not find data for the specified scenarios in the results file.")
        return

    df = pd.DataFrame(plot_data)

    # --- Generate the Plots ---
    sns.set_theme(style="whitegrid")
    color_palette = {
        "Baseline": "royalblue",
        "Best Intervention": "darkorange"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle('Distribution of Final Agent States by Intervention Scenario', fontsize=18, y=0.98)

    # Health Plot
    sns.violinplot(ax=axes[0], data=df[df['Metric'] == 'Health'], x='Scenario', y='value',
                   hue='Scenario', palette=color_palette, inner='quartile', legend=False, alpha=0.8)
    axes[0].set_title('Final Health Distribution', fontsize=14)
    axes[0].set_ylabel('State Value (1-100)', fontsize=12)
    axes[0].set_xlabel(None)

    # Wealth Plot
    sns.violinplot(ax=axes[1], data=df[df['Metric'] == 'Wealth'], x='Scenario', y='value',
                   hue='Scenario', palette=color_palette, inner='quartile', legend=False, alpha=0.8)
    axes[1].set_title('Final Wealth Distribution', fontsize=14)
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)
    
    # Set the y-axis limit based on the model's max state value
    max_val = SVEIRConfig().steering_parameters.max_state_value
    axes[0].set_ylim(0, max_val + 5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    output_dir = os.path.dirname(full_results_file)
    output_filename = os.path.join(output_dir, f"violin_plots_{experiment_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    
    plt.show()