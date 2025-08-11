# policy_computation/generator.py

import numpy as np
from scipy.stats import qmc
import torch

from dgl_ptm.config import SVEIRConfig
# Note the new import path for the engine
from .engine import value_iteration

def create_and_save_policy_library(config: SVEIRConfig, infection_risk_levels: np.ndarray):
    """
    Generates agent personas and their policy libraries for a given configuration
    and saves them to a compressed .npz file.

    Args:
        config (SVEIRConfig): The model configuration object.
        infection_risk_levels (np.ndarray): An array of the discretized risk
                                            levels to compute policies for.
    """
    # 1. Generate agent personas using Latin Hypercube Sampling (LHS)
    sampler = qmc.LatinHypercube(d=4, seed=config.seed)
    samples = sampler.random(n=config.num_agent_personas)
    param_ranges = [config.alpha_range, config.gamma_range, config.omega_range, config.eta_range]
    scaled_samples = qmc.scale(samples, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
    agent_personas = torch.from_numpy(scaled_samples).float()

    # 2. Pre-compute policies for each persona and risk level
    policy_library = {}
    params_for_vi = config.steering_parameters.model_dump()

    for persona_id in range(config.num_agent_personas):
        persona_policies = []
        alpha, gamma, omega, eta = agent_personas[persona_id]

        for risk_level in infection_risk_levels:
            params_for_vi['global_infection_prob'] = risk_level
            policy = value_iteration(
                max_state_value=100,
                alpha=alpha.item(),
                gamma=gamma.item(),
                theta=config.steering_parameters.theta,
                omega=omega.item(),
                eta=eta.item(),
                beta=config.steering_parameters.beta,
                params=params_for_vi
            )
            persona_policies.append(policy)
        policy_library[persona_id] = np.stack(persona_policies)
    
    # 3. Save the results to the specified file
    np.savez_compressed(
        config.policy_library_path,
        agent_personas=agent_personas.numpy(),
        infection_risk_levels=infection_risk_levels,
        **{f"policies_{pid}": policy_library[pid] for pid in policy_library}
    )
