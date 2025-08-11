"""Time-stepping module for the SVEIR model."""

from typing import Any, Dict

import torch
import dgl

from dgl_ptm.agent.agent_update import sveir_agent_update
from dgl_ptm.model.data_collection import data_collection

# Define compartment mapping globally or as a constant if it's fixed
COMPARTMENT_MAP = {
    "S": 0,  # Susceptible
    "V": 1,  # Vaccinated
    "E": 2,  # Exposed
    "I": 3,  # Infectious
    "R": 4   # Recovered
}

def _calculate_adjacency(agent_graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Calculates the adjacency matrix based on agent spatial coordinates.

    Two agents are considered adjacent if they share the exact same (x, y) coordinates.
    This is used to model infection transmission in a shared location.

    Args:
        agent_graph (dgl.DGLGraph): The DGL graph containing agent 'x' and 'y'
                                    coordinates in its node data.

    Returns:
        torch.Tensor: A square adjacency matrix where A[i,j] = 1 if agents i and j
                      are at the same location (and i != j), else 0.
    """
    coordinates = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']], dim=1)
    # Compare all pairs of coordinates. (N, 1, 2) == (1, N, 2) -> (N, N, 2)
    # .all(-1) checks if both x and y match -> (N, N) bool tensor
    # .float() converts bool to float (0.0 or 1.0)
    # .fill_diagonal_(0) sets self-loops to 0, as an agent cannot infect itself.
    adjacency_matrix = (coordinates.unsqueeze(1) == coordinates.unsqueeze(0)).all(dim=-1).float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def sveir_step(
    agent_graph: dgl.DGLGraph,
    device: torch.device,
    timestep: int,
    params: Dict[str, Any],
    grid: Any,
    policy_library: torch.Tensor,
    risk_levels
) -> None:
    """
    Performs a single step of the SVEIR model simulation.

    This function orchestrates the various updates that occur in one time step,
    including agent movement, disease progression, health investment decisions,
    and data collection. The order of these operations is critical for the
    simulation's logic.

    Args:
        agent_graph (dgl.DGLGraph): The DGLGraph representing agents and their states.
                                    This graph is modified in-place.
        device (torch.device): The computation device (e.g., 'cpu' or 'cuda').
        timestep (int): The current simulation time step.
        params (Dict[str, Any]): A dictionary of steering parameters for the model.
        grid (Any): The spatial grid environment object.
        optimal_policy (torch.Tensor): The pre-computed optimal policy table for
                                       health investment decisions.
    """
    # On the first step, collect initial state data.
    if timestep == 0:
        data_collection(
            agent_graph,
            timestep=timestep,
            npath=params['npath'],
            epath=params['epath'],
            ndata=params['ndata'],
            edata=params['edata'],
            mode=params['mode']
        )

    num_nodes = agent_graph.num_nodes()

    # --- DYNAMIC ENVIRONMENT UPDATE ---
    # Update the global infection probability for this timestep.
    # Here we draw from a normal distribution and clamp it to be non-negative.
    # A more complex model (e.g., seasonal sine wave) could be used here.
    current_infection_prob = torch.normal(mean=params["infection_prob_mean"], std=params["infection_prob_std"]).item()
    current_infection_prob = max(0.001, current_infection_prob)
    
    # Store it in the params dict to pass to agent_update
    params['infection_probability'] = current_infection_prob

    # Calculate edge weights for social interaction (e.g., visiting friends)
    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((num_nodes, num_nodes), device=device)
    edge_weights[src, dst] = agent_graph.edata["weight"].to(device)

    # --- Agent and Environment Updates ---
    
    # 1. Agent movement based on time use distribution
    random_activity = sveir_agent_update("move", agent_graph, edge_weights=edge_weights)

    # 2. Increment exposure time for agents in the 'Exposed' state
    sveir_agent_update("exposure_increment", agent_graph, M=COMPARTMENT_MAP)

    # 3. Transition from Exposed to Infectious after exposure period
    sveir_agent_update("exposed_to_infectious", agent_graph, M=COMPARTMENT_MAP, params=params)

    # 4. Transition from Infectious to Recovered based on recovery rate
    sveir_agent_update("infectious_to_recovered", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes)

    # 5. Transition from Susceptible to Vaccinated based on vaccination rate
    sveir_agent_update("susceptible_to_vaccinated", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes)

    # 6. Agent health investment decision and subsequent wealth/health updates
    sveir_agent_update("health_investment", agent_graph, params=params, policy_library=policy_library, risk_levels=risk_levels)

    # 7. Calculate adjacency based on current locations for infection transmission
    adjacency = _calculate_adjacency(agent_graph).to(device)

    # 8. Transmission from Susceptible to Exposed (human-to-human)
    sveir_agent_update("susceptible_to_exposed", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes,
                       adjacency=adjacency)

    # 9. Transmission from Vaccinated to Exposed (breakthrough infections)
    sveir_agent_update("vaccinated_to_exposed", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes,
                       adjacency=adjacency)

    # 10. Water-to-human transmission at contaminated water points
    sveir_agent_update("water_to_human_transmission", agent_graph, M=COMPARTMENT_MAP, params=params, grid=grid)

    # 11. Human-to-water transmission (infected agents contaminate water points)
    sveir_agent_update("human_to_water_transmission", agent_graph, M=COMPARTMENT_MAP, params=params, grid=grid,
                       random_activity=random_activity)

    # 12. Random recovery of contaminated water collection points
    sveir_agent_update("water_recovery", agent_graph, params=params, grid=grid)

    # 13. Cyclical shock event contaminates water points
    if (timestep + 1) % params["shock_frequency"] == 0:
        sveir_agent_update("shock", agent_graph, params=params, grid=grid)

    # --- Data Collection ---
    
    # Determine if data should be collected at this timestep
    do_periodical_data_collection = (
        params['data_collection_period'] > 0
        and (timestep % params['data_collection_period'] == 0)
    )
    do_specific_data_collection = (
        params['data_collection_list']
        and (timestep in params['data_collection_list'])
    )

    if do_periodical_data_collection or do_specific_data_collection:
        data_collection(
            agent_graph,
            timestep=timestep + 1,
            npath=params['npath'],
            epath=params['epath'],
            ndata=params['ndata'],
            edata=params['edata'],
            mode=params['mode']
        )