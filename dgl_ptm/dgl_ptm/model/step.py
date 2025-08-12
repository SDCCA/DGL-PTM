"""Time-stepping module for the SVEIR model."""

from typing import Any, Dict, Tuple

import torch
import dgl

from dgl_ptm.agent.agent_update import sveir_agent_update
from dgl_ptm.model.data_collection import data_collection

# Define compartment mapping globally or as a constant if it's fixed
COMPARTMENT_MAP = {
    "S": 0, "V": 1, "E": 2, "I": 3, "R": 4
}

def _calculate_adjacency(agent_graph: dgl.DGLGraph) -> torch.Tensor:
    """Calculates the adjacency matrix based on agent spatial coordinates."""
    coordinates = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']], dim=1)
    adjacency_matrix = (coordinates.unsqueeze(1) == coordinates.unsqueeze(0)).all(dim=-1).float()
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix

def sveir_step(
    agent_graph: dgl.DGLGraph,
    device: torch.device,
    timestep: int,
    params: Dict[str, Any],
    grid: Any,
    policy_library: dict,
    risk_levels: torch.Tensor
) -> Tuple[int, Dict[str, int]]:
    """
    Performs a single step of the SVEIR model simulation and returns key metrics.
    
    Returns:
        A tuple containing:
        - int: The number of newly infected agents in this step.
        - dict: A dictionary with the counts of agents in each compartment.
    """
    # if timestep == 0:
    #     data_collection(agent_graph, timestep=timestep, npath=params['npath'], epath=params['epath'], ndata=params['ndata'], edata=params['edata'], mode=params['mode'])

    num_nodes = agent_graph.num_nodes()
    
    # --- DYNAMIC ENVIRONMENT & AGENT UPDATES ---    
    mean_prob = params["infection_prob_mean"]
    std_prob = params["infection_prob_std"]
    params['infection_probability'] = max(0.001, torch.normal(mean=mean_prob, std=std_prob, size=(1,)).item())

    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((num_nodes, num_nodes), device=device)
    edge_weights[src, dst] = agent_graph.edata["weight"].to(device)

    random_activity = sveir_agent_update("move", agent_graph, edge_weights=edge_weights)
    sveir_agent_update("exposure_increment", agent_graph, M=COMPARTMENT_MAP)
    
    newly_infectious_count = sveir_agent_update("exposed_to_infectious", agent_graph, M=COMPARTMENT_MAP, params=params)
    
    sveir_agent_update("infectious_to_recovered", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes)
    sveir_agent_update("susceptible_to_vaccinated", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes)
    sveir_agent_update("health_investment", agent_graph, params=params, policy=policy_library, risk_levels=risk_levels)

    adjacency = _calculate_adjacency(agent_graph).to(device)
    sveir_agent_update("susceptible_to_exposed", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes, adjacency=adjacency)
    sveir_agent_update("vaccinated_to_exposed", agent_graph, M=COMPARTMENT_MAP, params=params, num_nodes=num_nodes, adjacency=adjacency)
    
    sveir_agent_update("water_to_human_transmission", agent_graph, M=COMPARTMENT_MAP, params=params, grid=grid)
    sveir_agent_update("human_to_water_transmission", agent_graph, M=COMPARTMENT_MAP, params=params, grid=grid, random_activity=random_activity)
    sveir_agent_update("water_recovery", agent_graph, params=params, grid=grid)
    if (timestep + 1) % params["shock_frequency"] == 0:
        sveir_agent_update("shock", agent_graph, params=params, grid=grid)

    # --- Data Collection ---
    if (params['data_collection_period'] > 0 and (timestep % params['data_collection_period'] == 0)) or \
       (params['data_collection_list'] and (timestep in params['data_collection_list'])):
        data_collection(
            agent_graph,
            timestep=timestep + 1,
            npath=params['npath'],
            epath=params['epath'],
            ndata=params['ndata'],
            edata=params['edata'],
            mode=params['mode']
        )

    # --- Tally compartment counts for time-series tracking ---
    compartments = agent_graph.ndata["compartments"]
    compartment_counts = {
        "S": torch.sum(compartments == COMPARTMENT_MAP["S"]).item(),
        "V": torch.sum(compartments == COMPARTMENT_MAP["V"]).item(),
        "E": torch.sum(compartments == COMPARTMENT_MAP["E"]).item(),
        "I": torch.sum(compartments == COMPARTMENT_MAP["I"]).item(),
        "R": torch.sum(compartments == COMPARTMENT_MAP["R"]).item(),
    }

    return newly_infectious_count, compartment_counts