#!/usr/bin/env python

"""step - time-stepping for the poverty-trap model."""

from dgl_ptm.agent.agent_update import agent_update
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

def sveir_step(agent_graph, device, timestep, params):
    """Step - time-stepping module for the SVEIR model.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents
        device: Device to run the model on, e.g. 'cpu' or 'cuda'
        timestep: Current time step
        params: List of user-defined parameters

    Output:
        agent_graph: Updated agent_graph after one step of functional manipulation
    """
    # map from compartment to index
    m = {
        "S":0,
        "V":1,
        "E":2,
        "I":3,
        "R":4
    }

    # random tensors
    bounds = [0.0, 1.0]
    num_nodes = agent_graph.num_nodes()
    num_edges = agent_graph.num_edges()
    i_r_rng = sample_distribution_tensor('uniform', bounds, num_nodes)
    s_r_rng = sample_distribution_tensor('uniform', bounds, num_nodes)

    # increment exposure time
    agent_graph.ndata["exposure_time"][agent_graph.ndata["compartments"]==m["E"]] += 1

    # transition from Exposed to Infectious
    exposed_to_infections = (agent_graph.ndata["compartments"]==m["E"]) & (agent_graph.ndata["exposure_time"] >= params["exposure_period"])
    agent_graph.ndata["compartments"][exposed_to_infections] = m["I"]

    # transition from Infectious to Recovered
    infectious_to_recovered = (agent_graph.ndata["compartments"]==m["I"]) & (i_r_rng < params["recovery_rate"])
    agent_graph.ndata["compartments"][infectious_to_recovered] = m["R"]

    # transition from Susceptible to Vaccinated
    susceptible_to_vaccinated = (agent_graph.ndata["compartments"]==m["S"]) & (s_r_rng < params["vaccination_rate"])
    agent_graph.ndata["compartments"][susceptible_to_vaccinated] = m["V"]

    # transition from Susceptible to Exposed
    src, dst = agent_graph.edges()
    edge_weights = torch.zeros((num_nodes, num_nodes))
    edge_weights[src, dst] = agent_graph.edata["weight"]
    susceptible_nodes = (agent_graph.ndata["compartments"] == 0).nonzero(as_tuple=True)[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, edge_weights, 0)[susceptible_nodes]
    nonzero_weights = torch.where(infected_weights > 0)

    RNG = torch.rand((4, nonzero_weights[0].shape[0]))
    RNG1 = torch.zeros_like(infected_weights)
    RNG1[nonzero_weights] = RNG[0]
    RNG2 = torch.zeros_like(infected_weights)
    RNG2[nonzero_weights] = RNG[1]

    infection = (infected_weights > 0).type(torch.float32) * (RNG1 < infected_weights) * (RNG2 < params["infection_probability"])
    infected_nodes = susceptible_nodes[torch.where(infection.sum(axis=1) > 0)]

    agent_graph.ndata["compartments"][infected_nodes] = m["E"]
    agent_graph.ndata["exposure_time"][infected_nodes] = 0

    # transition from Vaccinated to Exposed
    vaccinated_nodes = (agent_graph.ndata["compartments"] == m["V"]).nonzero(as_tuple=True)[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, edge_weights, 0)[vaccinated_nodes]
    nonzero_weights = torch.where(infected_weights > 0)

    RNG = torch.rand((2, nonzero_weights[0].shape[0]))
    RNG1 = torch.zeros_like(infected_weights)
    RNG1[nonzero_weights] = RNG[0]
    RNG2 = torch.zeros_like(infected_weights)
    RNG2[nonzero_weights] = RNG[1]

    infection = (infected_weights > 0).type(torch.float32) * (RNG1 < infected_weights) * (RNG2 < params["infection_probability"])
    infected_nodes = vaccinated_nodes[torch.where(infection.sum(axis=1) > 0)]

    agent_graph.ndata["compartments"][infected_nodes] = m["E"]
    agent_graph.ndata["exposure_time"][infected_nodes] = 0

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
