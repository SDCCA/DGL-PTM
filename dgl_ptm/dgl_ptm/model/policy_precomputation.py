"""
This module contains the logic for pre-computing and saving the policy library.
"""
import logging
import numpy as np
from scipy.stats import qmc
import torch
from tqdm import tqdm

from dgl_ptm.config import SVEIRConfig
from dgl_ptm.model.policy_engine import value_iteration

logger = logging.getLogger(__name__)

def generate_and_save_policy_library(config: SVEIRConfig):
    """
    Generates agent personas and policies, then saves them to a file.

    This function performs the computationally expensive value iteration and saves
    the results to the path specified in the configuration.

    Args:
        config (SVEIRConfig): The model configuration object.
    """
    # 1. Generate agent personas using LHS
    logger.info(f"Generating {config.num_agent_personas} agent personas using LHS.")
    sampler = qmc.LatinHypercube(d=4, seed=config.seed)
    samples = sampler.random(n=config.num_agent_personas)
    param_ranges = [config.alpha_range, config.gamma_range, config.omega_range, config.eta_range]
    scaled_samples = qmc.scale(samples, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
    agent_personas = torch.from_numpy(scaled_samples).float()

    # 2. Pre-compute policies for each (persona, p_h_decrease) pair
    num_p_values = len(config.p_h_decrease_values)
    total_policies = config.num_agent_personas * num_p_values
    logger.info(f"Pre-computing {total_policies} policies...")

    policy_library = {}
    for persona_id in range(config.num_agent_personas):
        persona_policies = []
        alpha, gamma, omega, eta = agent_personas[persona_id]

        for p_h_decrease_val in tqdm(config.p_h_decrease_values):
            policy = value_iteration(
                100,
                alpha.item(),
                gamma.item(),
                config.steering_parameters.theta,
                omega.item(),
                eta.item(),
                config.steering_parameters.beta,
                config.steering_parameters.P_H_increase,
                config.steering_parameters.wealth_update_A,
                p_h_decrease_val
            )
            persona_policies.append(policy)
        
        policy_library[persona_id] = np.stack(persona_policies)
    
    # 3. Save the results to a compressed .npz file
    # We save both the personas and the policies together.
    # We must use **kwargs to save the dictionary of policies correctly.
    logger.info(f"Saving policy library and personas to {config.policy_library_path}")
    np.savez_compressed(
        config.policy_library_path,
        agent_personas=agent_personas.numpy(),
        **{f"policies_{pid}": policy_library[pid] for pid in policy_library}
    )
    logger.info("Pre-computation complete and file saved successfully.")