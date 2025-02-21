#!/usr/bin/env python

"""step - time-stepping for the poverty-trap model."""

from dgl_ptm.agent.agent_update import agent_update, sveir_agent_update
from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.agentInteraction.weight_update import multi_property_weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.network.global_attachment import global_attachment
from dgl_ptm.network.link_deletion import link_deletion
from dgl_ptm.network.local_attachment import local_attachment
from dgl_ptm.network.local_attachment_basic_homophily import local_attachment_homophily
from dgl_ptm.network.random_edge_noise import random_edge_noise
from dgl_ptm.util.utils import sample_distribution_tensor
import torch

def ptm_step(agent_graph, device, timestep, params):
    """Step - time-stepping module for the poverty-trap model.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents
        device: Device to run the model on, e.g. 'cpu' or 'cuda'
        timestep: Current time step
        params: List of user-defined parameters

    Output:
        agent_graph: Updated agent_graph after one step of functional manipulation
    """
    if params['step_type']=='default':
        #Wealth transfer
        trade_money(agent_graph, device, method = params['trade_method'])

        #Edge manipulation
        local_attachment(
            agent_graph, n_FoF_links = 1, edge_prop = 'weight', p_attach=1.
            )
        link_deletion(
            agent_graph,
            method = params['del_method'],
            threshold = params['del_threshold']
            )
        global_attachment(agent_graph, device, ratio = params['noise_ratio'])

        #Update agent states
        agent_update(agent_graph, params, device=device)

        #Weight update
        multi_property_weight_update(
            agent_graph,
            device,
            truncation_weight = params['truncation_weight'],
            properties = {'wealth':{'keys':["wealth"],'homophily_parameter':params['homophily_parameter'],'characteristic_distance':params['characteristic_distance']}})
        
    elif params['step_type']=='ptm':
        if timestep==0:
            if agent_graph.number_of_edges()+params['noise_ratio']*agent_graph.number_of_nodes()+params['local_ratio']*agent_graph.number_of_nodes()<2**32:
                agent_graph = agent_graph.int()
                print(f"Agent graph storage type: {agent_graph.idtype}")



            #Update agent income
            agent_update(agent_graph,
                         params,
                         device=device,
                         method ='income'
                         )
            #Update agent consumption
            agent_update(
                agent_graph,
                params,
                device=device,
                timestep=timestep,
                method ='consumption'
            )
            #Collect specified data
            data_collection(
                agent_graph,
                timestep = timestep,
                npath = params['npath'],
                epath = params['epath'],
                ndata = params['ndata'],
                edata = params['edata'],
                mode = params['mode']
                )
            return
        #For timestep 1 and beyond:

        #Update agent capital, k_t+1 for the previous step becomes k_t
        agent_update(
            agent_graph, 
            params, 
            device=device, 
            timestep=timestep, 
            method = 'capital'
            )
        #Update agent theta with the information from the previous step
        agent_update(
            agent_graph, params, device=device, timestep=timestep-1, method ='theta'
            )

        #Update edge weights

        multi_property_weight_update(
            agent_graph,
            device,
            truncation_weight = params['truncation_weight'],
            properties = params['homophily_basis'])


        #Edge manipulation
        start_edges = agent_graph.number_of_edges()
        random_edge_noise(
            agent_graph,
            device,
            n_perturbances = int(params['noise_ratio']*agent_graph.number_of_nodes())
            )
        local_attachment_homophily(
            agent_graph,
            device,
            n_FoF_links = int(params['local_ratio']*agent_graph.number_of_nodes()),
            homophily_parameter = params['homophily_parameter'],
            characteristic_distance = params['characteristic_distance'],
            truncation_weight = params['truncation_weight']
            )
        if params['del_threshold'] == 'balance':
            threshold = int((agent_graph.number_of_edges()-start_edges)/2)
        else:
            threshold = params['del_threshold']
        link_deletion(
            agent_graph, method = params['del_method'], threshold = threshold
            )
        #Update agent degree and weighted degree
        agent_update(agent_graph, method='degree')
        agent_update(agent_graph, method='weighted_degree')

        #Wealth transfer
        trade_money(agent_graph, device, method = params['trade_method'])


        # Update agent income
        agent_update(agent_graph, params, device=device, method ='income')
        # Predict agent consumption (and investment if applicable)
        agent_update(
            agent_graph, params, device=device, timestep=timestep, method ='consumption'
            )


    # Data can be collected periodically (every X steps) and/or at specified time steps.
    do_periodical_data_collection = (
        params['data_collection_period'] > 0
        and timestep % params['data_collection_period'] == 0
        )
    do_specific_data_collection = (
        params['data_collection_list']
        and timestep in params['data_collection_list']
        )
    if do_periodical_data_collection or do_specific_data_collection:
        #Data collection and storage
        data_collection(
            agent_graph,
            timestep = timestep,
            npath = params['npath'],
            epath = params['epath'],
            ndata = params['ndata'],
            edata = params['edata'],
            mode = params['mode']
            )

def sveir_step(agent_graph, device, timestep, params, grid):
    """Step - time-stepping module for the SVEIR model.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents
        device: Device to run the model on, e.g. 'cpu' or 'cuda'
        timestep: Current time step
        params: List of user-defined parameters

    Output:
        agent_graph: Updated agent_graph after one step of functional manipulation
    """
    num_nodes = agent_graph.num_nodes()

    M = {
        "S":0,
        "V":1,
        "E":2,
        "I":3,
        "R":4
    }

    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((num_nodes, num_nodes))
    edge_weights[src, dst] = agent_graph.edata["weight"]

    # agents choose activity based on time use distribution
    random_activity = sveir_agent_update("move", agent_graph, edge_weights=edge_weights)

    # increment exposure time
    sveir_agent_update("exposure_increment", agent_graph, M)

    # Exposed -> Infectious
    sveir_agent_update("exposed_to_infectious", agent_graph, M, params)

    # Infectious -> Recovered
    sveir_agent_update("infectious_to_recovered", agent_graph, M, params, num_nodes)

    # Susceptible -> Vaccinated
    sveir_agent_update("susceptible_to_vaccinated", agent_graph, M, params, num_nodes)

    # Susceptible -> Exposed
    coordinates = torch.stack([agent_graph.ndata['x'], agent_graph.ndata['y']]).T
    adjacency = (coordinates[:, None, :] == coordinates[None, :, :]).all(-1).float().fill_diagonal_(0)
    sveir_agent_update("susceptible_to_exposed", agent_graph, M, params, num_nodes, edge_weights, adjacency=adjacency)

    # Vaccinated -> Exposed
    sveir_agent_update("vaccinated_to_exposed", agent_graph, M, params, num_nodes, edge_weights, adjacency=adjacency)

    # water -> human infection
    sveir_agent_update("water_to_human_transmission", agent_graph, M, params, grid=grid, random_activity=random_activity)

    # human -> water infection
    sveir_agent_update("human_to_water_transmission", agent_graph, M, params, grid=grid, random_activity=random_activity)

    sveir_agent_update("water_recovery", agent_graph, params=params, grid=grid)

    # Data can be collected periodically (every X steps) and/or at specified time steps.
    do_periodical_data_collection = (
        params['data_collection_period'] > 0
        and timestep % params['data_collection_period'] == 0
    )
    do_specific_data_collection = (
        params['data_collection_list']
        and timestep in params['data_collection_list']
    )
    if do_periodical_data_collection or do_specific_data_collection:
        data_collection(
            agent_graph,
            timestep = timestep,
            npath = params['npath'],
            epath = params['epath'],
            ndata = params['ndata'],
            edata = params['edata'],
            mode = params['mode']
        )
